import copy
import math
from typing import List, Union

import numpy as np
import torch
from ferret.evaluators.faithfulness_measures import TauLOO_Evaluation
from ferret.model_utils import ModelHelper
from tqdm.autonotebook import tqdm
from transformers.tokenization_utils_base import BatchEncoding


class TauLOO_Evaluation_For_Regression(TauLOO_Evaluation):

    def __init__(
        self, model, tokenizer, OHE=None, specs=None, specifiersep=None, tokensep=None
    ):
        self.helper = RegressionModelHelper(model, tokenizer)
        self.OHE = OHE
        self.specs = specs
        self.specifiersep = specifiersep
        self.tokensep = tokensep

    def compute_leave_one_out_occlusion(
        self,
        text,
        id=None,
        remove_first_last=True,
        handle_tokens="remove",
        scaler=None,
        batch_size=8,
        replacement_dict=None,
        replacespecifier=False,
        dev=False,
    ):

        if dev:
            text = text[:200]
        _, logits = self.helper._forward(
            text=text,
            OHE=self.OHE,
            specs=None if self.specs is None else self.specs[id][None, :, :],
            batch_size=batch_size,
            add_special_tokens=self.OHE is None,
        )
        if logits.size(-1) == 1:
            pred = logits[:, 0].cpu().numpy()
        else:
            pred = logits.softmax(-1)[:, 0].cpu().numpy()

        item = self.helper._tokenize(text, add_special_tokens=self.OHE is None)
        input_len = item["attention_mask"].sum().item()
        input_ids = item["input_ids"][0][:input_len].tolist()
        if remove_first_last == True and self.OHE is None:
            input_ids = input_ids[1:-1]
            input_len -= 2

        samples = list()
        sample_specs = list()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        _tokens = list()
        if self.specs is not None:
            for t, s in zip(tokens, self.specs[id]):
                if sum(s) == 0.0:
                    _tokens.append(t)
                else:
                    _tokens.append(
                        t + self.specifiersep + self.specifiersep.join(map(str, s))
                    )
            tokens = _tokens

        replacements = None
        if handle_tokens == "replace":
            replacements = []

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        mask_token_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            mask_token_id = pad_token_id

        for occ_idx in range(input_len):
            sample = copy.copy(input_ids)
            if handle_tokens == "replace":
                occ_token = self.tokenizer.convert_ids_to_tokens(
                    sample[occ_idx]
                ).replace("Ġ", "")
                if occ_token in replacement_dict:
                    replace_list = replacement_dict[occ_token]
                else:
                    replace_list = []
                if self.specs is not None:
                    _replace_list = []
                    for t in replace_list:
                        if sum(self.specs[id][occ_idx]) == 0.0:
                            _replace_list.append(t)
                        else:
                            _replace_list.append(
                                t
                                + self.specifiersep
                                + self.specifiersep.join(
                                    map(str, self.specs[id][occ_idx])
                                )
                            )
                else:
                    _replace_list = replace_list
                replacements.append(_replace_list)
                replace_ids = [
                    self.tokenizer.convert_tokens_to_ids(x) for x in replace_list
                ]
                for r in replace_ids:
                    sample[occ_idx] = r
                    decoded_sample = self.tokenizer.decode(sample)
                    if self.tokensep is not None:
                        decoded_sample = decoded_sample.replace(" ", self.tokensep)
                    samples.append(decoded_sample)
                    if self.specs is not None:
                        sample_specs.append(self.specs[id])
                if replacespecifier and sum(self.specs[id][occ_idx]) > 0:
                    if self.OHE is not None:
                        decoded_sample = self.tokenizer.decode(sample)
                        if self.tokensep is not None:
                            decoded_sample = decoded_sample.replace(" ", self.tokensep)
                        samples.append(decoded_sample)
                    else:
                        samples.append(text)
                    knockout_spec = self.specs[id].copy()
                    knockout_spec[occ_idx] = 0.0
                    sample_specs.append(knockout_spec)
                    replacements[-1].append(occ_token)
            else:
                if handle_tokens == "remove":
                    sample.pop(occ_idx)
                    if pad_token_id is not None:
                        sample.append(pad_token_id)
                elif handle_tokens == "mask" and mask_token_id is not None:
                    sample[occ_idx] = mask_token_id

                decoded_sample = self.tokenizer.decode(sample)
                if self.tokensep is not None:
                    decoded_sample = decoded_sample.replace(" ", self.tokensep)
                samples.append(decoded_sample)

        _, logits = self.helper._forward(
            text=samples,
            OHE=self.OHE,
            specs=(
                None
                if self.specs is None
                else (
                    np.tile(self.specs[id], (len(samples), 1, 1))
                    if not replacespecifier
                    else sample_specs
                )
            ),
            batch_size=batch_size,
            add_special_tokens=self.OHE is None,
        )
        if logits.size(-1) == 1:
            leave_one_out_removal = logits[:, 0].cpu()
        else:
            leave_one_out_removal = logits.softmax(-1)[:, 0].cpu()

        if scaler is None:
            occlusion_importance = leave_one_out_removal - pred
        else:
            pred = scaler.inverse_transform(pred)
            leave_one_out_removal = scaler.inverse_transform(leave_one_out_removal)
            occlusion_importance = leave_one_out_removal - pred

        return occlusion_importance, pred, tokens, replacements


class RegressionModelHelper(ModelHelper):

    def _forward(
        self,
        text: Union[str, List[str]],
        OHE=None,
        specs=None,
        batch_size=8,
        show_progress=False,
        use_input_embeddings=False,
        output_hidden_states=False,
        **tok_kwargs,
    ):
        if isinstance(text, str):
            text = [text]

        n_batches = math.ceil(len(text) / batch_size)
        batches = np.array_split(text, n_batches)

        outputs = list()
        with torch.no_grad():

            if show_progress:
                pbar = tqdm(total=n_batches, desc="Batch", leave=False)

            for batch in batches:
                item = self._tokenize(
                    batch.tolist(), padding="max_length", **tok_kwargs
                )
                if OHE is not None:
                    transformed = [
                        OHE.transform(np.reshape(x, (-1, 1))) for x in item["input_ids"]
                    ]

                    target_length = getattr(
                        getattr(self.model, "config", None),
                        "max_position_embeddings",
                        None,
                    )
                    if target_length is None:
                        target_length = max(t.shape[0] for t in transformed)

                    def _pad_sequence(arr, length):
                        cur_len = arr.shape[0]
                        if cur_len < length:
                            pad_width = ((0, length - cur_len), (0, 0))
                            return np.pad(
                                arr,
                                pad_width,
                                mode="constant",
                                constant_values=0.0,
                            )
                        if cur_len > length:
                            return arr[:length]
                        return arr

                    padded_inputs = [
                        _pad_sequence(arr, int(target_length)).astype(np.float32)
                        for arr in transformed
                    ]

                    if specs is not None:
                        specs_list = [np.asarray(spec) for spec in specs]
                        adjusted_specs = [
                            (
                                _pad_sequence(spec_arr, int(target_length))
                                if spec_arr.shape[0] != int(target_length)
                                else spec_arr
                            )
                            for spec_arr in specs_list
                        ]
                        specs_array = np.stack(adjusted_specs, axis=0)
                        padded_inputs = [
                            np.concatenate((spec_arr, input_arr), axis=-1)
                            for spec_arr, input_arr in zip(specs_array, padded_inputs)
                        ]

                    item["input_ids"] = torch.tensor(
                        np.stack(padded_inputs, axis=0), dtype=torch.float
                    )
                item = {k: v.to(self.model.device) for k, v in item.items()}

                if use_input_embeddings:
                    ids = item.pop("input_ids")  # (B,S,d_model)
                    input_embeddings = self._get_input_embeds_from_ids(ids)
                    out = self.model(
                        inputs_embeds=input_embeddings,
                        **item,
                        output_hidden_states=output_hidden_states,
                    )
                else:
                    out = self.model(**item, output_hidden_states=output_hidden_states)
                outputs.append(out)

                if show_progress:
                    pbar.update(1)

        if show_progress:
            pbar.close()

        logits = torch.cat([o["logits"] for o in outputs])
        return outputs, logits
