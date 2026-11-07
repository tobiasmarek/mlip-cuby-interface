## TL;DR

### Keywords:
- `zmq` tested only minimally - but should be faster than http


### Backends:
#### Fairchem
- For the models to work you need to have a license
- Authentication needed `hf auth login`

#### MACE
- The models are very memory hungry for bigger structures
- Higher number of CPUs and memory size is recommended

#### MLatom
- Licensed package was required for AIQM3
- Semiempirical part fails to converge on some structures, raised etemp and lowered acc is sometimes needed

#### So3lr
- So3lr CLI interface is safer to use instead
- Can be used with a special displaced interaction protocol (`interaction_displace_separate_charge`, available on request)
- Fragment-wise charges are required in the displaced structure (extxyz passed instead of xyz)