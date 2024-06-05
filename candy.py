import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()

    def forward(self, x):
        return (torch.abs(x + 1) - torch.abs(x - 1)) / 2


custom_activation = CustomActivation().to(device)


class CANDY(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, in_channel, out_channel):
        super(CANDY, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.batch_size = batch_size

        # Initialize the p_mask hyperparameter
        self.p_mask = nn.Parameter(
            torch.randn(hidden_size, input_size), requires_grad=True
        )
        # layer 1: p set output
        self.p_output_layer = nn.Sequential(
            CustomActivation(), nn.Linear(input_size, input_size), CustomActivation()
        )
        self.Wp = nn.Parameter(
            torch.tril(torch.randn(hidden_size, hidden_size)), requires_grad=True
        )
        self.Wp.data.diagonal().fill_(1)
        self.Wp_diag = nn.Parameter(torch.ones(hidden_size), requires_grad=True)

        # layer 2: z set output
        self.z_output_layer = nn.Sequential(
            CustomActivation(), nn.Linear(input_size, input_size), CustomActivation()
        )
        self.Wzp = nn.Parameter(
            torch.randn(hidden_size, hidden_size), requires_grad=True
        )

        # MLP layer
        self.fc1 = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU(True))
        self.fc2 = nn.Linear(input_size, input_size)

    def split_input(self, x):
        p_mask = custom_activation(self.p_mask)
        p_set = x * p_mask

        return p_set

    def forward(self, input):
        in_channel = self.in_channel
        out_channel = self.out_channel
        input_size = self.input_size
        hidden_size = self.hidden_size
        batch_size = self.batch_size

        test_input = input.to(device)
        test_input = test_input.view(-1, in_channel, hidden_size, input_size)
        if test_input.shape[0] != batch_size:
            batch_size = int(test_input.shape[0])

        input = input.view(batch_size, in_channel, hidden_size, input_size)
        out = torch.Tensor(batch_size, hidden_size, input_size).to(device)
        x = torch.Tensor(in_channel, hidden_size, input_size).to(device)

        for i in range(batch_size):
            x = input[i]
            x = x / 5000
            residual = x.to(device)

            p_set = self.split_input(x)

            with torch.no_grad():
                self.Wp.data = torch.tril(self.Wp.data)
                self.Wp.data.diagonal().clamp_(min=0, max=1)
            Wp = self.Wp + torch.diag(self.Wp_diag).to(device)
            Wzp = self.Wzp

            p_output = torch.Tensor(out_channel, hidden_size, input_size).to(device)
            z_output = torch.Tensor(out_channel, hidden_size, input_size).to(device)
            for ii in range(in_channel):
                temp_p_output = torch.mm(Wp, p_set[ii])
                temp_p_output = self.p_output_layer(temp_p_output)
                temp_z_output = torch.mm(Wzp, temp_p_output)
                temp_z_output = self.z_output_layer(temp_z_output)

                p_output += temp_p_output.clone()
                z_output += temp_z_output.clone()

            combined_output = p_output + z_output + residual[0]
            output = self.fc1(combined_output)
            output = output.view(hidden_size, input_size)
            output = self.fc2(output)

            out[i] = output

        if batch_size < self.batch_size:
            out

        return out


class Multi_CANDY(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, in_channel, out_channel):
        super(Multi_CANDY, self).__init__()

        self.candy1 = CANDY(
            batch_size, input_size, hidden_size, in_channel, out_channel
        )
        self.candy2 = CANDY(
            batch_size, input_size, hidden_size, in_channel, out_channel
        )
        self.candy3 = CANDY(
            batch_size, input_size, hidden_size, in_channel, out_channel
        )

    def forward(self, x):
        if x.mean() < 2000:
            output = self.candy1(x)
        elif x.mean() > 6000:
            output = self.candy2(x)
        else:
            output = self.candy3(x)

        return output


class Cross_CANDY(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, in_channel, out_channel):
        super(Multi_CANDY, self).__init__()

        self.candy1 = CANDY(
            batch_size, input_size, hidden_size, in_channel, out_channel
        )
        self.candy2 = CANDY(
            batch_size, input_size, hidden_size, in_channel, out_channel
        )
        self.candy3 = CANDY(
            batch_size, input_size, hidden_size, in_channel, out_channel
        )
        self.candy4 = CANDY(
            batch_size, input_size, hidden_size, in_channel, out_channel
        )

    def forward(self, x):
        if x.mean() < 2000:
            output = self.candy1(x)
        elif x.mean() > 6000:
            output = self.candy2(x)
        else:
            candy3_input = torch.where(x > 4000, x, torch.tensor(0.0, device=x.device))
            candy4_input = torch.where(x <= 4000, x, torch.tensor(0.0, device=x.device))

            candy3_output = self.candy3(candy3_input)
            candy4_output = self.candy4(candy4_input)

            output = candy3_output + candy4_output

        return output
