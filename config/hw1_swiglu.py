# HW1 Q4 variant: swiglu. Same as train_shakespeare_char.py, one change only.
exec(open('config/train_shakespeare_char.py').read())
out_dir = 'out-hw1-swiglu'
compile = False # torch.compile needs triton, which isn't available on Windows/MPS
mlp_type = 'swiglu'
