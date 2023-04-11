import pandas as pd
import numpy as np
from sklearn.base import clone
from skorch.hf import HuggingfacePretrainedTokenizer
from transformers import BeitImageProcessor
from sklearn.model_selection import KFold


class CrossEmbeddings:
    """
    A class for generating multi-modal embeddings via cross-fitting.

    Attributes:
    -------
        dataset (pandas.DataFrame): the input dataset.
        text_col (str): the name of the text column in `dataset`.
        image_col (str): the name of the image column in `dataset`.
        d_col (str): the name of the treatment column in `dataset`.
        y_col (str): the name of the target column in `dataset`.
        n_folds (int): the number of folds for cross-validation.
        aux_d (class of dml_emb.FeatureRegressor):
                the auxiliary model for the treatment.
        aux_y (class of dml_emb.FeatureRegressor):
                the auxiliary model for the target.
        txt_str (str): the name of the HuggingFace
                       transformer for tokenizing text.
        img_str (str): the name of the HuggingFace
                       transformer for processing images.
        txt_tokens (dict): the tokenized text.
        img_features (dict): the image features.
        d (numpy.ndarray): the treatment array.
        y (numpy.ndarray): the target array.
        emb (dict): the embeddings for each fold.

    Methods:
    -------
        __init__(self, dataset, text_col, image_col,
                 d_col, y_col, n_folds, aux_d, aux_y, txt_str, img_str):
            Initializes the CrossEmbeddings class.
        _tokenizer(self, data, modelname):
            Tokenizes the text using the
            specified HuggingFace transformer.
        _feature_extractor(self, data, modelname):
            Extracts features from the images using the
            specified HuggingFace transformer.
        _prepare_data(self):
            Prepares the data for generating embeddings.
        slice_input(self, idx):
            Slices the input data for a given index.
        fit_and_predict_embeddings(self):
            Fits the auxiliary models for each fold
            and generates the embeddings.
        get_embedded_df(self):
            Returns the input dataset with the
            embeddings added as a new column.
        get_embeddings(self):
            Returns the embeddings as a numpy array.
    """

    def __init__(
        self,
        dataset,
        text_col,
        image_col,
        d_col,
        y_col,
        n_folds,
        aux_d,
        aux_y,
        txt_str,
        img_str,
    ):
        self.dataset = dataset.reset_index()
        self.text_col = text_col
        self.image_col = image_col
        self.d_col = d_col
        self.y_col = y_col
        self.n_folds = n_folds
        self.aux_d = aux_d
        self.aux_y = aux_y
        self.txt_str = txt_str
        self.img_str = img_str

        self.txt_tokens = None
        self.img_features = None
        self.d = None
        self.y = None
        self.emb = {fold: None for fold in range(self.n_folds)}

    def _tokenizer(self, data, modelname: str):
        txt = data[self.text_col]
        hf_tokenizer = HuggingfacePretrainedTokenizer(modelname, max_length=128)
        hf_tokenizer.fit(txt)
        return hf_tokenizer.transform(txt)

    def _feature_extractor(self, data, modelname: str):
        hf_extractor = BeitImageProcessor.from_pretrained(modelname)
        return hf_extractor(list(data[self.image_col]), return_tensors="pt")

    def _prepare_data(self):
        self.txt_tokens = self._tokenizer(self.dataset, self.txt_str)
        self.img_features = self._feature_extractor(self.dataset, self.img_str)

    def slice_input(self, idx):
        inputs = {
            "input_ids": self.txt_tokens["input_ids"][idx],
            "attention_mask": self.txt_tokens["attention_mask"][idx],
            "features": self.img_features["pixel_values"][idx],
        }
        return inputs

    def fit_and_predict_embeddings(self):
        self._prepare_data()
        for fold, (train_idx, output_idx) in enumerate(
            KFold(n_splits=self.n_folds, shuffle=True, random_state=1234).split(
                self.dataset
            )
        ):
            print(f"Fold {fold+1} from {self.n_folds}")
            aux_y_tmp = clone(self.aux_y)
            aux_d_tmp = clone(self.aux_d)
            y_tmp = (
                self.dataset.loc[train_idx, self.y_col]
                .to_numpy()
                .reshape(-1, 1)
                .astype(np.float32)
            )
            d_tmp = (
                self.dataset.loc[train_idx, self.d_col]
                .to_numpy()
                .reshape(-1, 1)
                .astype(np.float32)
            )
            temp_inputs = self.slice_input(train_idx)
            print(" Fit target aux model")
            aux_y_tmp.fit(temp_inputs, y_tmp)
            y_emd_tmp = aux_y_tmp.predict_features(self.slice_input(output_idx))
            del aux_y_tmp
            print(" Fit treatment aux model")
            aux_d_tmp.fit(temp_inputs, d_tmp)
            d_emd_tmp = aux_d_tmp.predict_features(self.slice_input(output_idx))
            del aux_d_tmp
            emb_tmp = np.concatenate((d_emd_tmp, y_emd_tmp), axis=1)
            self.emb[fold] = {idx: emb_tmp[x] for x, idx in enumerate(output_idx)}
        print("Finished")

    def get_embedded_df(self):
        d = {}
        for f in range(len(self.emb)):
            d = {**d, **self.emb[f]}
        self.dataset["emb"] = self.dataset.index.map(d)
        return self.dataset.set_index("index")

    def get_embeddings(self):
        return np.stack(self.dataset["emb"].to_numpy())
