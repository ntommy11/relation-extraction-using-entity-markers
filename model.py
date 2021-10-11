import torch

from transfomers import AutoModel, AutoTokenizer


class StartTokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = "klue/roberta-large"
        self.bert_model = AutoModel.from_pretrained(self.MODEL_NAME)
        self.hidden_size = 1024
        self.num_labels = 30
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

        special_tokens_dict = {
            "additional_special_tokens": [
                "[SUB:ORG]",
                "[SUB:PER]",
                "[/SUB]",
                "[OBJ:DAT]",
                "[OBJ:LOC]",
                "[OBJ:NOH]",
                "[OBJ:ORG]",
                "[OBJ:PER]",
                "[OBJ:POH]",
                "[/OBJ]",
            ]
        }

        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
        print("num_added_tokens:", num_added_tokens)

        self.bert_model.resize_token_embeddings(len(self.tokenizer))

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2 * self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size, self.num_labels),
        )

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        sub_token_index,
        obj_token_index,
    ):
        out = self.bert_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        h = out.last_hidden_state
        batch_size = h.shape[0]

        stack = []

        for i in range(batch_size):
            stack.append(
                torch.cat([h[i][sub_token_index[i]], h[i][obj_token_index[i]]])
            )
        stack = torch.stack(stack)

        out = self.classifier(stack)
        return out
