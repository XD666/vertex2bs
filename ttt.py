

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.xx = nn.Linear(128,64)
        self.vertice_map_r = nn.Linear(64, 16)
    def forward(self, x):
        out = self.vertice_map_r(self.xx(x))


model = Model(args)
model.load_state_dict(torch.load(args.pretrained_path))
model = model.to(torch.device(args.device))
for param in model.parameters():
    param.requires_grad = False

numb_fes = model.vertice_map_r.in_features
model.vertice_map_r = nn.Sequential(
    nn.Linear(numb_fes, 51),
)

Then train the model and save the trained model as save.pth

how to load the trained model using load_state_dict().
