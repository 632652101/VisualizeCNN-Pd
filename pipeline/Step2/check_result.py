from reprod_log import ReprodDiffHelper

if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("result_torch.npy")
    paddle_info = diff_helper.load_info("result_paddle.npy")

    diff_helper.compare_info(torch_info, paddle_info)

    diff_helper.report(diff_method="mean", path="pipeline/Step2/result_diff.log")
