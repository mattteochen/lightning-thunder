import torch.nn as nn
import thunder
import torch

model = nn.Sequential(nn.Linear(2048, 4096, dtype=torch.float16), nn.ReLU(), nn.Linear(4096, 64, dtype=torch.float16))

from thunder.recipes import BaseRecipe

r = BaseRecipe(interpreter="thunder.jit", fuser="torch.compile")
r.executor_names = ["torch"]

thunder_model = thunder.compile(model, recipe=r)
x = torch.randn(64, 2048, dtype=torch.float16)
report = thunder.estimate(thunder_model, x, type="memory", strategy="last_step")
print(report)
