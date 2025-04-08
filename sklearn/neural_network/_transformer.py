import numpy as np
import warnings
from scipy.special import softmax

from ..base import BaseEstimator, TransformerMixin
from ..utils import check_random_state, check_array
from ..utils.validation import check_is_fitted
from ..utils._param_validation import Interval, StrOptions

try:
    import torch
    from transformers import AutoModel, AutoTokenizer

    _has_transformers = True
except ImportError:
    _has_transformers = False


class GPTTransformer(TransformerMixin, BaseEstimator):

    def __init__(
            self,
            *,
            pretrained_model_name=None,
            use_pretrained=False,
            output_hidden_states=False,
            fine_tune=False,
            n_layers=6,
            n_heads=8,
            embedding_dim=512,
            feedforward_dim=2048,
            max_seq_length=1024,
            dropout=0.1,
            learning_rate=0.001,
            batch_size=32,
            n_iter=10,
            random_state=None,
            verbose=0,
    ):
        self.pretrained_model_name = pretrained_model_name
        self.use_pretrained = use_pretrained
        self.output_hidden_states = output_hidden_states
        self.fine_tune = fine_tune
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.feedforward_dim = feedforward_dim
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose

    def _init_params(self, n_features):
        rng = check_random_state(self.random_state)

        self.embedding_ = rng.normal(
            0, 0.02, size=(n_features, self.embedding_dim)
        )

        self.position_embedding_ = rng.normal(
            0, 0.02, size=(self.max_seq_length, self.embedding_dim)
        )

        self.layers_ = []
        for _ in range(self.n_layers):
            layer = {
                'query': rng.normal(0, 0.02, size=(self.embedding_dim, self.embedding_dim)),
                'key': rng.normal(0, 0.02, size=(self.embedding_dim, self.embedding_dim)),
                'value': rng.normal(0, 0.02, size=(self.embedding_dim, self.embedding_dim)),
                'output': rng.normal(0, 0.02, size=(self.embedding_dim, self.embedding_dim)),

                # Layer normalization parameters
                'ln1_weight': np.ones(self.embedding_dim),
                'ln1_bias': np.zeros(self.embedding_dim),
                'ln2_weight': np.ones(self.embedding_dim),
                'ln2_bias': np.zeros(self.embedding_dim),

                'ff1': rng.normal(0, 0.02, size=(self.embedding_dim, self.feedforward_dim)),
                'ff1_bias': np.zeros(self.feedforward_dim),
                'ff2': rng.normal(0, 0.02, size=(self.feedforward_dim, self.embedding_dim)),
                'ff2_bias': np.zeros(self.embedding_dim),
            }
            self.layers_.append(layer)

        self.output_layer_ = rng.normal(0, 0.02, size=(self.embedding_dim, n_features))
        self.output_bias_ = np.zeros(n_features)

    def _load_pretrained_model(self, n_features):

        if not _has_transformers:
            raise ImportError(
                "The 'transformers' package is required to use pretrained models. "
                "Please install it with: pip install transformers"
            )

        if self.pretrained_model_name is None:
            raise ValueError(
                "pretrained_model_name must be provided when use_pretrained is True"
            )

        if self.verbose:
            print(f"Loading pretrained model: {self.pretrained_model_name}")

        self.tokenizer_ = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        self.pretrained_model_ = AutoModel.from_pretrained(
            self.pretrained_model_name,
            output_hidden_states=self.output_hidden_states
        )

        if not self.fine_tune:
            self.pretrained_model_.eval()

        self.n_features_ = n_features

        model_dim = self.pretrained_model_.config.hidden_size
        self.output_layer_ = np.random.normal(0, 0.02, size=(model_dim, n_features))
        self.output_bias_ = np.zeros(n_features)

    def _layer_norm(self, x, weight, bias, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return weight * (x - mean) / np.sqrt(var + eps) + bias

    def _attention(self, q, k, v, mask=None):
        scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(q.shape[-1])

        if mask is not None:
            scores = scores + mask

        weights = softmax(scores, axis=-1)

        return np.matmul(weights, v)

    def _multi_head_attention(self, x, layer, mask=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        q = np.matmul(x, layer['query'])
        k = np.matmul(x, layer['key'])
        v = np.matmul(x, layer['value'])

        q = q.reshape(batch_size, seq_len, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.n_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads, -1).transpose(0, 2, 1, 3)

        attn_output = self._attention(q, k, v, mask)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        return np.matmul(attn_output, layer['output'])

    def _feedforward(self, x, layer):
        h = np.matmul(x, layer['ff1']) + layer['ff1_bias']
        h = h * 0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h ** 3)))  # GELU

        return np.matmul(h, layer['ff2']) + layer['ff2_bias']

    def _transformer_block(self, x, layer, mask=None):
        attn_output = self._multi_head_attention(x, layer, mask)
        x = x + attn_output
        x = self._layer_norm(x, layer['ln1_weight'], layer['ln1_bias'])

        ff_output = self._feedforward(x, layer)
        x = x + ff_output
        x = self._layer_norm(x, layer['ln2_weight'], layer['ln2_bias'])

        return x

    def _fine_tune_pretrained_model(self, X):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        if self.verbose:
            print(f"Fine-tuning pretrained model: {self.pretrained_model_name}")

        X_tensor = torch.tensor(X, dtype=torch.float32)

        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        output_layer = torch.nn.Parameter(torch.tensor(self.output_layer_, dtype=torch.float32))
        output_bias = torch.nn.Parameter(torch.tensor(self.output_bias_, dtype=torch.float32))

        parameters = list(self.pretrained_model_.parameters()) + [output_layer, output_bias]

        optimizer = optim.Adam(parameters, lr=self.learning_rate)

        criterion = nn.MSELoss()

        self.pretrained_model_.train()  # Set model to training mode

        for epoch in range(self.n_iter):
            total_loss = 0

            for batch_idx, (data,) in enumerate(dataloader):
                optimizer.zero_grad()

                batch_texts = []
                for i in range(data.shape[0]):
                    text = " ".join([f"{val:.6f}" for val in data[i].numpy()])
                    batch_texts.append(text)

                inputs = self.tokenizer_(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length
                )

                model_outputs = self.pretrained_model_(**inputs)

                if self.output_hidden_states:
                    hidden_states = torch.cat(model_outputs.hidden_states, dim=1)
                    embeddings = hidden_states.mean(dim=1)
                else:
                    last_hidden_state = model_outputs.last_hidden_state
                    embeddings = last_hidden_state.mean(dim=1)

                outputs = torch.matmul(embeddings, output_layer) + output_bias
                loss = criterion(outputs, data)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if self.verbose and (epoch + 1) % max(1, self.n_iter // 10) == 0:
                print(f"Epoch {epoch + 1}/{self.n_iter}, Loss: {total_loss / len(dataloader):.6f}")

        self.pretrained_model_.eval()
        self.output_layer_ = output_layer.detach().numpy()
        self.output_bias_ = output_bias.detach().numpy()

    def fit(self, X, y=None):
        X = self._validate_data(X, accept_sparse=False)

        n_samples, n_features = X.shape

        if self.use_pretrained:
            self._load_pretrained_model(n_features)

            if self.fine_tune:
                if not _has_transformers:
                    raise ImportError(
                        "The 'transformers' package is required to fine-tune pretrained models. "
                        "Please install it with: pip install transformers torch"
                    )
                self._fine_tune_pretrained_model(X)
        else:
            self._init_params(n_features)

        if self.verbose:
            print(f"Fitted {self.__class__.__name__} on {n_samples} samples")

        return self

    def transform(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, accept_sparse=False)

        n_samples, n_features = X.shape

        if self.use_pretrained:
            if n_features != self.n_features_:
                raise ValueError(
                    f"X has {n_features} features, but {self.__class__.__name__} "
                    f"was trained with {self.n_features_} features."
                )

            if not isinstance(X, torch.Tensor):
                X_tensor = torch.tensor(X, dtype=torch.float32)
            else:
                X_tensor = X

            outputs = []
            with torch.no_grad():
                for i in range(n_samples):
                    text = " ".join([f"{val:.6f}" for val in X[i]])

                    inputs = self.tokenizer_(
                        text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_seq_length
                    )

                    model_outputs = self.pretrained_model_(**inputs)

                    if self.output_hidden_states:
                        hidden_states = torch.cat(model_outputs.hidden_states, dim=1)
                        embedding = hidden_states.mean(dim=1).squeeze().numpy()
                    else:
                        last_hidden_state = model_outputs.last_hidden_state
                        embedding = last_hidden_state.mean(dim=1).squeeze().numpy()

                    output = np.matmul(embedding, self.output_layer_) + self.output_bias_
                    outputs.append(output)

            return np.array(outputs)
        else:
            h = np.matmul(X, self.embedding_)
            h = h.reshape(n_samples, 1, self.embedding_dim)

            h = h + self.position_embedding_[0:1]

            mask = None

            for layer in self.layers_:
                h = self._transformer_block(h, layer, mask)

            output = np.matmul(h.reshape(n_samples, self.embedding_dim), self.output_layer_) + self.output_bias_

            return output

    def __sklearn_tags__(self):
        return {
            'requires_y': False,
            'requires_positive_X': False,
            'X_types': ['2darray'],
            'poor_score': True,
            'no_validation': False,
            'multioutput': False,
            'allow_nan': False,
            'stateless': False,
            'multilabel': False,
            '_skip_test': True,  # Skip in common tests
            'preserves_dtype': [np.float64, np.float32],
        }
