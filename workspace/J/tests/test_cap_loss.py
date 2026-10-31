import torch
import torch.nn.functional as F

from trainer.losses import asu_cap_loss


class Frozen(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.stored = logits

    def forward(self, input_ids=None, labels=None, attention_mask=None, **kwargs):
        from transformers.modeling_outputs import CausalLMOutputWithPast

        zero = self.stored.reshape(-1)[0] * 0
        return CausalLMOutputWithPast(loss=zero, logits=self.stored)


def test_cap_is_zero_when_student_gold_is_below_teacher():
    torch.manual_seed(0)
    labels = torch.tensor([[-100, -100, 1, 2]])
    m_tar = torch.tensor([[0, 0, 1, 1]])
    student = torch.zeros(1, 4, 4)
    teacher = torch.zeros(1, 4, 4)
    student[0, 1, 1] = 5
    teacher[0, 1, 1] = 8
    student[0, 2, 2] = 1
    teacher[0, 2, 2] = 4
    student = student.requires_grad_(True)
    loss = asu_cap_loss(
        Frozen(student),
        Frozen(teacher),
        [(torch.zeros(1, 4, dtype=torch.long), labels, torch.ones(1, 4), m_tar)],
        attention_temp=2.3,
        layers_id=None,
    )
    assert float(loss) == 0.0


def test_cap_positive_only_on_m_tar_when_student_exceeds_teacher():
    labels = torch.tensor([[-100, 3]])
    m_tar = torch.tensor([[0, 1]])
    student = torch.zeros(1, 2, 4, requires_grad=True)
    teacher = torch.zeros(1, 2, 4)
    with torch.no_grad():
        student[0, 0, 3] = 4
        teacher[0, 0, 3] = 1
    loss = asu_cap_loss(
        Frozen(student),
        Frozen(teacher),
        [(torch.zeros(1, 2, dtype=torch.long), labels, torch.ones(1, 2), m_tar)],
    )
    loss.backward()
    log_ps = F.log_softmax(student[0, 0], dim=-1)[3]
    log_pt = F.log_softmax(teacher[0, 0], dim=-1)[3]
    assert abs(float(loss) - float(torch.relu(log_ps - log_pt))) < 1e-5
    assert student.grad[0, 0, 3] > 0
