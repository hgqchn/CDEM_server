import sys
import os

import torch
import torch.nn.functional as F
import utils
import dem_utils


# 评估bic dem 和sr dem的质量

if __name__ == '__main__':

    hr_file=r'test_data/hr.TIF'

    bic_file=r'test_data/bic.TIF'
    sr_file=r'test_data/sr.TIF'

    hr_dem = utils.read_dem(hr_file)
    hr_4dtensor = torch.from_numpy(hr_dem).unsqueeze(0).unsqueeze(0)

    sr_dem = utils.read_dem(sr_file)
    sr_4dtensor = torch.from_numpy(sr_dem).unsqueeze(0).unsqueeze(0)

    bic_dem = utils.read_dem(bic_file)
    bic_4dtensor = torch.from_numpy(bic_dem).unsqueeze(0).unsqueeze(0)

    bic_eval_res= dem_utils.cal_DEM_metric(bic_4dtensor, hr_4dtensor, padding=1)
    bic_eval_str=utils.compose_kwargs(**bic_eval_res)
    print(f"bicubic dem: {bic_eval_str}")

    sr_eval_res= dem_utils.cal_DEM_metric(sr_4dtensor, hr_4dtensor, padding=1)
    sr_eval_str=utils.compose_kwargs(**sr_eval_res)
    print(f"sr dem: {sr_eval_str}")

    pass
