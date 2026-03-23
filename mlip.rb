################################################################################
#
# MLIP interface
#
# Author: Marek Tobias, Martin Novacek, Jan Rezac
# Date created: 2020-03-11
# License: Cuby4 license
# Description: Interface to Machine Learning Interatomic Potentials (MLIPs)
# Status: Works
#
################################################################################

require "json"
require "open3"
require "socket"
require "timeout"

module InterfaceMlip
	#=======================================================================
	# Interface header
	#=======================================================================
	DEVELOPMENT_FLAG = :warning
	DEVELOPMENT_STATUS = "Manager-only Ruby interface with persistent Python workers"
	INTERFACE = :calculation_external
	CAPABILITIES = [:energy, :gradient]
	MODIFIER = true
	DIRECTORY = "MLIP"
	METHODS = {
		:"mlip" => [:energy, :gradient],
	}

	# Added settings:
	DEFAULT_TIMEOUT = 15 * 60
	
	# Server Class Variables
	@@mlip_worker_id = nil
    @@mlip_port = nil

	# Bridge Class Variables
    @@mlip_bridge_in = nil
    @@mlip_bridge_out = nil
    @@mlip_bridge_err = nil
    @@mlip_bridge_wait = nil
    @@mlip_bridge_err_thread = nil
	#=======================================================================

	def prepare_interface
		@mlip_model_path = @settings[:mlip_model].to_s.strip # Check if file exists? (wouldn't handle model names)
		@mlip_backend = @settings[:mlip_backend].to_s.strip.downcase.to_sym
		@mlip_transport = @settings[:mlip_server].to_s.strip.downcase

		if @@mlip_worker_id
            return # Check whether worker is already running
        end

		if @mlip_transport == "zmq"
			mlip_check_python_module("zmq")
		end
		
		@@mlip_port = mlip_find_free_port(8000, 100)

		begin
			@@mlip_worker_id = Process.spawn(*mlip_worker_command)
			mlip_start_bridge
			mlip_wait_for_worker
		rescue
			InterfaceMlip.final_cleanup
			raise
		end
	end

	def calculate_interface
		return mlip_single_atom_result if @settings[:mlip_set_atom_to_zero] && @geometry.size == 1

		worker_out = mlip_bridge_request(
			{
				"cmd" => "calculate",
				"xyz" => mlip_geometry_to_xyz,
				"gradients" => @what.include?(:gradient),
				"charge" => @settings[:charge],
			},
			DEFAULT_TIMEOUT
		)
		mlip_build_results(worker_out)
	end

	def cleanup_interface
		# Nothing to do in-between calculations
	end

	#=======================================================================
	# Private methods
	#=======================================================================

	def mlip_worker_command
		device = @settings[:mlip_device].to_s.strip
		device = "auto" if device == ""

		command = [
			"python3",
			"#{interface_dir}/mlip_worker_server.py",
			"--transport", @mlip_transport,
			"--host", "127.0.0.1",
			"--port", @@mlip_port.to_s,
			"--backend", @mlip_backend.to_s,
			"--model", @mlip_model_path,
			"--device", device,
		]

		command << "--sp-only" if @settings[:mlip_sp_only]

		cpu_threads = @settings[:mlip_cpu_threads].to_i
		command += ["--cpu-threads", cpu_threads.to_s] if cpu_threads > 0

		cuda_fraction = @settings[:mlip_cuda_memory_fraction].to_f
		command += ["--cuda-memory-fraction", cuda_fraction.to_s] if cuda_fraction > 0.0

		command
	end

	def mlip_start_bridge
		@@mlip_bridge_in, @@mlip_bridge_out, @@mlip_bridge_err, @@mlip_bridge_wait = Open3.popen3(
			"python3",
			"#{interface_dir}/mlip_bridge_client.py",
			"--transport", @mlip_transport,
			"--host", "127.0.0.1",
			"--port", @@mlip_port.to_s
		)

		@@mlip_bridge_err_thread = Thread.new do
			begin
				@@mlip_bridge_err.each_line { |line| $stderr.puts(line) unless line.strip == "" }
			rescue IOError
			end
		end
	end

	def mlip_wait_for_worker
		ready = false
		600.times do
			begin
				ready = (mlip_bridge_request({ "cmd" => "ping" }, 2) == "OK")
			rescue
				ready = false
			end
			break if ready
			sleep(0.5)
		end

		unless ready
			InterfaceMlip.final_cleanup
			Cuby::error "MLIP interface: Worker does not respond"
		end
	end

	def mlip_bridge_request(payload, timeout_s)
		unless @@mlip_bridge_in && @@mlip_bridge_out
			Cuby::error "MLIP bridge is not initialized"
		end

		@@mlip_bridge_in.puts(payload.to_json)
		@@mlip_bridge_in.flush

		line = nil
		begin
			Timeout.timeout(timeout_s) { line = @@mlip_bridge_out.gets }
		rescue Timeout::Error
			Cuby::error "MLIP request timeout after #{timeout_s} seconds"
		end

		if line.nil?
			Cuby::error "MLIP bridge terminated unexpectedly"
		end

		out = JSON.parse(line)
		unless out["status"] == "ok"
			Cuby::error "MLIP bridge error: #{out['error']}"
		end
		out["result"]
	end

	def mlip_build_results(worker_out)
		results = Results.new
		results.energy = worker_out["energy"].to_f * @settings[:mlip_multiplier]
		return results unless @what.include?(:gradient)

		forces = worker_out["forces"]
		unless forces && forces.size == @geometry.size
			Cuby::error "MLIP worker returned invalid forces payload"
		end

		results.gradient = Gradient.new
		@geometry.each_index do |i|
			x = forces[i][0].to_f * -1.0
			y = forces[i][1].to_f * -1.0
			z = forces[i][2].to_f * -1.0
			results.gradient << Coordinate[x, y, z] * @settings[:mlip_multiplier]
		end
		results
	end

	def mlip_single_atom_result
		results = Results.new
		results.energy = 0.0
		if @what.include?(:gradient)
			results.gradient = Gradient.new
			results.gradient << Coordinate[0, 0, 0]
		end
		results
	end

	def mlip_geometry_to_xyz
		lines = [@geometry.size.to_s, "Generated by InterfaceMLIP"]
		@geometry.each do |atom|
			x, y, z = atom.to_a
			lines << "%2s % .15f % .15f % .15f" % [atom.element.to_s, x, y, z]
		end
		lines.join("\n") + "\n"
	end

	def mlip_find_free_port(start_port, max_tries)
		port = start_port
		max_tries.times do
			begin
				server = TCPServer.new("127.0.0.1", port)
				server.close
				return port
			rescue Errno::EADDRINUSE, Errno::EACCES
				port += 1
			end
		end
		
		Cuby::error "No free port found for MLIP worker after #{max_tries} attempts starting from port #{start_port}"
	end

	def mlip_check_python_module(mod_name)
		_stdout, _stderr, status = Open3.capture3("python3", "-c", "import #{mod_name}")
		unless status.success?
			Cuby::error "Missing Python module '#{mod_name}' for transport '#{@mlip_transport}'. Install it with: python3 -m pip install ..."
		end
	end

	#=======================================================================
	# Static methods
	#=======================================================================

	@@mlip_sentinel = Object.new # Dummy object
    ObjectSpace.define_finalizer(@@mlip_sentinel, proc { |id| 
        InterfaceMlip.final_cleanup
    })

	def self.final_cleanup
        # 1. Shutdown worker gracefully
        if @@mlip_bridge_in && @@mlip_bridge_out && !@@mlip_bridge_in.closed?
            begin
                @@mlip_bridge_in.puts({ "cmd" => "shutdown" }.to_json)
                @@mlip_bridge_in.flush
            rescue
                # Ignore broken pipes if worker already died
            end
        end
		

        # 2. Stop bridge client
        if @@mlip_bridge_wait
            @@mlip_bridge_in.close unless @@mlip_bridge_in.closed? rescue nil
            @@mlip_bridge_out.close unless @@mlip_bridge_out.closed? rescue nil
            @@mlip_bridge_err.close unless @@mlip_bridge_err.closed? rescue nil

            begin
                Process.kill("TERM", @@mlip_bridge_wait.pid)
            rescue Errno::ESRCH
            end

            if @@mlip_bridge_err_thread
                @@mlip_bridge_err_thread.kill rescue nil
            end
        end

        # 3. Stop worker process
        if @@mlip_worker_id
            begin
                Process.kill("TERM", @@mlip_worker_id)
                # Use WNOHANG (non-blocking wait) so the finalizer doesn't freeze
                Process.wait(@@mlip_worker_id, Process::WNOHANG)
            rescue Errno::ESRCH, Errno::ECHILD
            end
            @@mlip_worker_id = nil
        end

        # Reset variables
        @@mlip_bridge_in = nil
        @@mlip_bridge_out = nil
        @@mlip_bridge_err = nil
        @@mlip_bridge_wait = nil
        @@mlip_bridge_err_thread = nil
    end
end
