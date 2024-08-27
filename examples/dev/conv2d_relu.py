import torch
import thunder


class Module(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1) -> None:
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        a = self.conv2d(x)
        b = self.conv2d(x)
        c = self.conv2d(x + x)
        d = self.relu(b * a)
        return c + d


with torch.device("cuda"):
    model = Module(16, 33, 3, stride=2)
    x = torch.randn(20, 16, 50, 100)

    jmodel = thunder.jit(model)

    ans = jmodel(x)
    # print('---------------------------------------------- all traces')
    # for t in thunder.last_traces(jmodel):
    #     print(t)
    #     print('##############################################')
    # print('---------------------------------------------- ans')
    # print(ans)
