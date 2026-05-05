from hatchery.core.examples.train_sft import shifted_completion_labels, tokenize_example


class FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        assert tokenize is False
        rendered = ""
        for message in messages:
            rendered += f"<{message['role']}>{message['content']}"
        if kwargs.get("add_generation_prompt"):
            rendered += "<assistant>"
        return rendered

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]


def test_shifted_completion_labels_pre_shift_targets():
    full_ids = [10, 11, 12, 13, 14]
    input_ids, labels = shifted_completion_labels(full_ids, prompt_len=3)

    assert input_ids == [10, 11, 12, 13]
    assert labels == [-100, -100, 13, 14]


def test_tokenize_example_uses_tinker_pre_shift_convention():
    tok = FakeTokenizer()

    item = tokenize_example(tok, "Apple falls.")
    input_ids = item["model_input"]["chunks"][0]["tokens"]
    labels = item["loss_fn_inputs"]["target_tokens"]["data"]

    assert len(input_ids) == len(labels)
    assert labels.count(-100) > 0
    first_trained_position = next(i for i, label in enumerate(labels) if label != -100)
    assert labels[first_trained_position] != input_ids[first_trained_position]
