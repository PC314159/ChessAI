import time

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *
from torch.utils.data import DataLoader, random_split

full_data = torch.load("../data/dataset_id_1")

train_size = int(len(full_data) * 0.8)
test_size = len(full_data) - train_size

train_data, test_data = random_split(full_data, [train_size, test_size])

print("len of train_data", len(train_data))
print("len of test data", len(test_data))

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

am = FENAnalyzerModel(num_res_blocks=11)
checkpoint = torch.load(
    "../output_models/2024/version_8-1_sgd_11_res_blocks_with_decay/Chess_FEN_Analyzer_Model_Round_70.pth")
am.load_state_dict(checkpoint, strict=False)
am = am.cuda()

loss_fn = nn.MSELoss()
loss_fn = loss_fn.cuda()

learning_rate = 0.001
weight_decay_const = 0.002
optimizer = torch.optim.SGD(params=am.parameters(), lr=learning_rate, weight_decay=weight_decay_const)

total_train_step = 475000
total_test_step = 70
resume_epoch = 70
epoch = 200


writer = SummaryWriter("../logs_train")
start_time = time.time()

for i in range(resume_epoch,epoch):
    print(f"Round {i + 1} starting")

    for data in train_dataloader:
        values, targets = data
        values, targets = values.cuda(), targets.cuda()
        outputs = am(values)
        outputs = outputs.squeeze()
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 1000 == 0:
            end_time = time.time()
            print(f"{end_time - start_time} since start time")
            print(f"Training {total_train_step}, Loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    total_test_loss = 0
    total_accurate = 0
    with torch.no_grad():
        for data in test_dataloader:
            values, targets = data
            values, targets = values.cuda(), targets.cuda()
            outputs = am(values)
            outputs = outputs.squeeze()
            loss = loss_fn(outputs, targets)

            total_test_loss += loss
            accurate = (torch.abs(torch.sub(outputs, targets)) <= 0.2).sum()
            total_accurate += accurate

    print(f"Test loss: {total_test_loss}")
    print(f"Total Accurate: {total_accurate}")
    print(f"Total Accuracy: {total_accurate / len(test_data)}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("total_accuracy", total_accurate / len(test_data), total_test_step)
    total_test_step += 1

    torch.save(am.state_dict(), f"../output_models/Chess_FEN_Analyzer_Model_Round_{i + 1}.pth")

writer.close()
