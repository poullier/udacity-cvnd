import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        # convert words into a vector
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # lstm no dropout, one layer and 512 embed size
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # convert the hidden state output dimension to the vocab_size
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        # remove end words in each caption
        embeds = self.embedding(captions[:, :-1])

        # insert a dimension of size one at position 1
        features = features.unsqueeze(1)

        # concat features and embeds
        inputs = torch.cat((features, embeds), 1)

        # pass to lstm
        hidden, states = self.lstm(inputs)
        output = self.linear(hidden)
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        words = []
        for i in range(max_len):
            hidden, states = self.lstm(inputs, states)
            outputs = self.linear(hidden.squeeze(1))
            idx = outputs.argmax(1)
            word = idx.item()
            words.append(word)
            # reach end
            if word == 1:
                break
            # use predicted word as input of the next iteration
            inputs = self.embedding(idx.unsqueeze(1))
        return words