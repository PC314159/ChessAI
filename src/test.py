import torch
import torchvision.transforms
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader

from src.model import FENAnalyzerModel


def main():
    full_data = torch.load("../data/dataset_id_1")

    test_dataloader = DataLoader(full_data, batch_size=1)

    am = FENAnalyzerModel(num_filter_channels=256)
    checkpoint = torch.load(
        "../output_models/2025/eval_move_models/v6/Chess_FEN_Analyzer_Model_Round_23.pth")
    am.load_state_dict(checkpoint, strict=False)
    am = am.cuda()

    am.eval()
    with (torch.no_grad()):
        for data in test_dataloader:
            values, target_eval, target_move = data
            values, target_eval, target_move = values.float().cuda(), target_eval.float().cuda(), target_move.long().cuda()
            eval_output, move_output = am(values)
            eval_output = eval_output.squeeze(-1)
            # move_output = move_output.squeeze()
            print("--------")
            print(eval_output)
            movel = []
            for i in range(len(move_output)):
                row = move_output[i]
                idx = row.argmax().item()
                movel.append((idx//64//8, idx//64%8, idx%64//8, idx%64%8))
            print(movel)

            loss = nn.MSELoss()(eval_output, target_eval)
            loss2 = -((move_output * target_move).sum(dim=1) + 0.00001).log().mean()

            eval_accurate = (torch.abs(torch.sub(eval_output, target_eval)) <= 0.5).sum()
            move_accurate = ((move_output * target_move) >= 0.1).sum()
            print(loss)
            print(loss2)
            print(eval_accurate)
            print(move_accurate)

if __name__ == "__main__":
    main()