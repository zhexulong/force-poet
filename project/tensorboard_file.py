import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

# Specify the path to the existing event file
original_event_file = "/home/sebastian/repos/poet/train/2024-09-16_09:10:40/events.out.tfevents.1726563347.asterix.31871.0"

# Specify the path for the new event file
new_event_file = "/home/sebastian/repos/poet/train/2024-09-16_09:10:40/events.out.tfevents.1726563347.asterix.31871.0_new"

# Load the existing event file using EventAccumulator
event_acc = event_accumulator.EventAccumulator(original_event_file)
event_acc.Reload()
tags = event_acc.Tags()["scalars"]

# Create a new event file and write the logged values into it
writer = SummaryWriter(log_dir=new_event_file)

for tag in tags:
    value_list = event_acc.Scalars(tag)

    # Rename tags if necessary
    if tag == "Train/loss_trans":
        new_tag = "Train/position_loss"
    elif tag == "Train/loss_rot":
        new_tag = "Train/rotation_loss"
    else:
        new_tag = tag

    # Write the new values with the potentially updated tag
    for value in value_list:
        if new_tag == "Train/position_loss":
            writer.add_scalar(new_tag, value.value / 2, global_step=value.step)
        else:
            writer.add_scalar(new_tag, value.value, global_step=value.step)

# Close the writer to finalize the new event file
writer.close()