from datasets import load_dataset

def check_endmodule_condition():
    dataset = load_dataset("yangyiyao/HaVen-KL-Dataset")
    examples = dataset["train"]
    invalid_indices = []
    output_endmodule_cnt = dict()
    instr_cnt = 0

    for idx, example in enumerate(examples):
        instruction = example['instruction']
        responses = example['output']

        # response는 리스트로 되어 있으므로 마지막 응답만 사용
        if not responses:
            continue  # 비어있는 응답은 무시
        # response = responses[-1]

        instr_has_endmodule = "endmodule" in instruction
        response_endmodule_count = responses.count("endmodule")
        if output_endmodule_cnt.get(response_endmodule_count):
            output_endmodule_cnt[response_endmodule_count] += 1
        else:
            output_endmodule_cnt[response_endmodule_count] = 1
        
        if instr_has_endmodule:
            instr_cnt+=1

        if instr_has_endmodule or response_endmodule_count != 1:
            invalid_indices.append(idx)

    print(f"{len(examples)} 중에, 총 {len(invalid_indices)}개의 샘플이 조건을 만족하지 않음.")
    # print("조건을 만족하지 않는 인덱스:", invalid_indices)
    print(f"endmodule in instruction: {instr_cnt}, endmodule in output: {output_endmodule_cnt}")

check_endmodule_condition()