# HW1 Q4 variant: gqa. Same as train_shakespeare_char.py, one change only.
exec(open('config/train_shakespeare_char.py').read())
out_dir = 'out-hw1-gqa'
compile = False # torch.compile needs triton, which isn't available on Windows/MPS
n_kv_head = 3 # 6 query heads, group size 2 -> 3 kv heads
