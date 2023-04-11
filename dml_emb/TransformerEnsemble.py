import torch
from torch import nn
from transformers import BeitForImageClassification
from transformers import BertForSequenceClassification


class FineTuned_TransformerEnsemble(nn.Module):
    """
    A PyTorch module that fine-tunes pre-trained image and text transformers and combines their outputs.

    Args:
    -------
        image_model (str): An image transformer model name from Huggingface transformers.
        text_model (str): A text transformer model name from the Huggingface transformers.
        num_labels (int): The number of output labels for classification (num_labels = 1 for regression tasks).

    Attributes:
    -------
        image_model (transformers): The pre-trained image transformer model.
        text_model (transformers): The pre-trained text transformer model.
        dropout (nn.Dropout): A dropout layer with a rate of 0.1.
        relu (nn.ReLU): A ReLU activation function.
        pre_classifier (nn.LazyLinear): A fully connected layer with output size 768.
        feature_out (nn.LazyLinear): A fully connected layer with output size 128.
        classifier (nn.LazyLinear): A fully connected layer with output size 1.

    Methods:
    -------
        forward(input_ids, attention_mask, features):
            Feeds the input through the image and text models and combines their outputs.

            Args:
                input_ids (torch.Tensor): A tensor of input token IDs for the text model, with shape (batch_size, sequence_length).
                attention_mask (torch.Tensor): A tensor of attention masks for the text model, with shape (batch_size, sequence_length).
                features (torch.Tensor): A tensor of input image features for the image model, with shape (batch_size, channels, heigt, width).

            Returns:
                logits (torch.Tensor): A tensor of output logits for classification, with shape (batch_size, num_labels).
                features (torch.Tensor): A tensor of output features, with shape (batch_size, 128).
    """

    def __init__(self, image_model, text_model, num_labels):
        super().__init__()
        self.image_model = BeitForImageClassification.from_pretrained(
            image_model, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        self.text_model = BertForSequenceClassification.from_pretrained(
            text_model,
            num_labels=num_labels,
        )
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.pre_classifier = nn.LazyLinear(768)
        self.feature_out = nn.LazyLinear(128)
        self.classifier = nn.LazyLinear(1)

    def forward(self, input_ids, attention_mask, features):
        text_features = self.text_model.bert(input_ids, attention_mask)
        image_features = self.image_model.beit(features)
        output_text = text_features[1]  # pooler_output
        output_image = image_features[1]  # pooler_output
        output_cat = torch.cat([output_text, output_image], dim=1)
        output_cat = self.pre_classifier(output_cat)
        output_cat = self.relu(output_cat)
        pooled_output = self.dropout(output_cat)
        features = self.feature_out(pooled_output)
        logits = self.classifier(features)
        return logits, features


class TransformerEnsemble_from_pretrained(nn.Module):
    """
    A PyTorch module that uses already fine-tuned and pre-trained
    image and text transformers and combines their outputs.

    Args:
    -------
        image_model (transformers): The pre-trained and fine-tuned image transformer model.
        text_model (transformers): The pre-trained and fine-tuned text transformer model.

    Attributes:
    -------
        image_model (transformers): The pre-trained and fine-tuned image transformer model.
        text_model (transformers): The pre-trained and fine-tuned text transformer model.
        dropout (nn.Dropout): A dropout layer with a rate of 0.1.
        relu (nn.ReLU): A ReLU activation function.
        pre_classifier (nn.LazyLinear): A fully connected layer with output size 768.
        feature_out (nn.LazyLinear): A fully connected layer with output size 128.
        classifier (nn.LazyLinear): A fully connected layer with output size 1.

    Methods:
    -------
        forward(input_ids, attention_mask, features):
            Feeds the input through the image and text models and combines their outputs.

            Args:
                input_ids (torch.Tensor): A tensor of input token IDs for the text model, with shape (batch_size, sequence_length).
                attention_mask (torch.Tensor): A tensor of attention masks for the text model, with shape (batch_size, sequence_length).
                features (torch.Tensor): A tensor of input image features for the image model, with shape (batch_size, channels, heigt, width).

            Returns:
                logits (torch.Tensor): A tensor of output logits for classification, with shape (batch_size, num_labels).
                features (torch.Tensor): A tensor of output features, with shape (batch_size, 128).
    """

    def __init__(self, image_model, text_model):
        super().__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.pre_classifier = nn.LazyLinear(768)
        self.feature_out = nn.LazyLinear(128)
        self.classifier = nn.LazyLinear(1)

    def forward(self, input_ids, attention_mask, features):
        with torch.no_grad():
            text_features = self.text_model(input_ids, attention_mask)
            image_features = self.image_model(features)
        output_text = text_features[1]
        output_image = image_features[1]
        output_cat = torch.cat([output_text, output_image], dim=1)
        output_cat = self.pre_classifier(output_cat)
        output_cat = self.relu(output_cat)
        pooled_output = self.dropout(output_cat)
        features = self.feature_out(pooled_output)
        logits = self.classifier(features)
        return logits, features
