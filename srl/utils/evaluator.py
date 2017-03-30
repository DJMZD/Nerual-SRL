def calculate_accuracy(errors):
    error_tag_num = 0.
    total_tag_num = 0.
    for sent in errors:
        error_tag_num += sum(sent)
        total_tag_num += len(sent)

    return 1 - error_tag_num / total_tag_num


def calculate_f_value(predicts, golds, tag_dict):
    class SRLSpan(object):
        def __init__(self, tag, begin_pos, end_pos):
            self.tag = tag
            self.begin_pos = begin_pos
            self.end_pos = end_pos

    def get_spans(tags):
        spans = []
        span = SRLSpan("", -1, -1)
        for i, tag_id in enumerate(tags):
            tag = tag_dict.convert(int(tag_id))
            if tag.startswith("B-"):
                # add previous span except verb
                if span.tag and span.tag != "V":
                    spans.append(span)
                span = SRLSpan(tag[2:], i, i)
            elif tag.startswith("I-"):
                # there's a span before
                if span.tag:
                    # the same tag, then change end position
                    if tag[2:] == span.tag:
                        span.end_pos = i
                    # otherwise add previous span(except verb)
                    else:
                        if span.tag != "V":
                            spans.append(span)
                        span = SRLSpan(tag[2:], i, i)
                else:
                    span = SRLSpan(tag[2:], i, i)
            else:
                # add previous span except verb
                if span.tag and span.tag != "V":
                    spans.append(span)
                # clear the span
                span = SRLSpan("", -1, -1)
        if span.tag:
            spans.append(span)

        return spans

    def has_span(spans, span):
        for s in spans:
            if (s.tag == span.tag and
                s.begin_pos == span.begin_pos and
                s.end_pos == span.end_pos):
                return True

        return False

    correct_tag_num = 0.
    predict_tag_num = 0.
    gold_tag_num = 0.
    for p, g in zip(predicts, golds):
        predict_spans = get_spans(p)
        gold_spans = get_spans(g)

        for p_s in predict_spans:
            if has_span(gold_spans, p_s) and not p_s.tag.startswith("C"):
                correct_tag_num += 1
        for p_s in predict_spans:
            if not p_s.tag.startswith("C"):
                predict_tag_num += 1
        for g_s in gold_spans:
            if not g_s.tag.startswith("C"):
                gold_tag_num += 1
    if predict_tag_num > 0:
        precision = correct_tag_num / predict_tag_num
    else:
        precision = 0.
    if gold_tag_num > 0:
        recall = correct_tag_num / gold_tag_num
    else:
        recall = 0.
    if precision + recall > 0:
        f_value = 2 * precision * recall / (precision + recall)
    else:
        f_value = 0.

    return (correct_tag_num, predict_tag_num, gold_tag_num,
            precision, recall, f_value)
