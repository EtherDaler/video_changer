from raft import RAFT
from utils.utils import InputPadder
import torch.nn.functional as F

def load_raft(model_path):
    model = RAFT()
    model.load_state_dict(torch.load(model_path))
    model = model.cuda().eval()
    return model

raft_model = load_raft("raft-things.pth")


def compute_flow_raft(model, frame1, frame2):

    frame1 = torch.from_numpy(frame1).permute(2,0,1).float()[None].cuda()
    frame2 = torch.from_numpy(frame2).permute(2,0,1).float()[None].cuda()

    with torch.no_grad():
        _, flow = model(frame1, frame2, iters=20, test_mode=True)

    flow = flow[0].permute(1,2,0).cpu().numpy()
    return flow