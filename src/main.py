import os
import random
import time

import numpy
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *
from torch.utils.data import DataLoader, random_split, ConcatDataset

data_dir = "../data/datasets_2024-01_no_draw_triple_3"
files = os.listdir(data_dir)
sampled_files = random.sample(files, 10)
datasets = [torch.load(os.path.join(data_dir, fname)) for fname in sampled_files]
full_data = ConcatDataset(datasets)

train_size = int(len(full_data) * 0.8)
test_size = len(full_data) - train_size

train_data, test_data = random_split(full_data, [train_size, test_size])

print("len of train_data", len(train_data))
print("len of test data", len(test_data))

train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=128, drop_last=True)

am = FENAnalyzerModel(num_res_blocks=11, num_filter_channels=256)
am = am.cuda()

loss_fn = nn.MSELoss()
loss_fn = loss_fn.cuda()

learning_rate = 0.0005
weight_decay_const = 0.0001
momentum_const = 0.9
l2a = 0.5
optimizer = torch.optim.SGD(params=am.parameters(), lr=learning_rate, weight_decay=weight_decay_const, momentum=momentum_const)

total_train_step = 0
total_test_step = 0
epoch = 50

writer = SummaryWriter("../logs_train/2025/eval_move_models/v7")
start_time = time.time()

for i in range(epoch):
    print(f"Round {i + 1} starting")

    for data in train_dataloader:
        values, target_eval, target_move = data
        values, target_eval, target_move = values.float().cuda(), target_eval.float().cuda(), target_move.long().cuda()
        eval_output, move_output = am(values)
        eval_output = eval_output.squeeze()
        move_output = move_output.squeeze()
        loss = loss_fn(eval_output, target_eval)
        loss2 = -((move_output * target_move).sum(dim=1) + 0.00001).log().mean()

        optimizer.zero_grad()
        (loss + l2a * loss2).backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 1000 == 0:
            end_time = time.time()
            print(f"{end_time - start_time} since start time")
            print(f"Training {total_train_step}, Loss1: {(loss).item()}")
            print(f"Training {total_train_step}, Loss2: {(loss2).item()}")
            if total_train_step % 1000 == 0:
                writer.add_scalar("train_loss1", (loss).item(), total_train_step)
                writer.add_scalar("train_loss2", (loss2).item(), total_train_step)

    total_test_loss = 0
    total_eval_accurate = 0
    total_move_accurate = 0
    with torch.no_grad():
        for data in test_dataloader:
            values, target_eval, target_move = data
            values, target_eval, target_move = values.float().cuda(), target_eval.float().cuda(), target_move.long().cuda()
            eval_output, move_output = am(values)
            eval_output = eval_output.squeeze()
            move_output = move_output.squeeze()
            loss = loss_fn(eval_output, target_eval)
            loss2 = -((move_output * target_move).sum(dim=1) + 0.00001).log().mean()

            total_test_loss += loss + l2a * loss2
            eval_accurate = (torch.abs(torch.sub(eval_output, target_eval)) <= 0.5).sum()
            total_eval_accurate += eval_accurate
            move_accurate = ((move_output * target_move) >= 0.1).sum()
            total_move_accurate += move_accurate

    print(f"Test loss: {total_test_loss}")
    print(f"Total Eval Accurate: {total_eval_accurate}")
    print(f"Total Eval Accuracy: {total_eval_accurate / len(test_data)}")
    print(f"Total Move Accurate: {total_move_accurate}")
    print(f"Total Move Accuracy: {total_move_accurate / len(test_data)}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("total_eval_accuracy", total_eval_accurate / len(test_data), total_test_step)
    writer.add_scalar("total_move_accuracy", total_move_accurate / len(test_data), total_test_step)
    total_test_step += 1


    torch.save(am.state_dict(), f"../output_models/2025/eval_move_models/v7/Chess_FEN_Analyzer_Model_Round_{i + 1}.pth")

writer.close()
