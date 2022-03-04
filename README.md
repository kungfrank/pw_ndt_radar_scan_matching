# pw_ndt_radar_scan_matching

The implementation of scanning radar scan matching proposed in ["A Normal Distribution Transform-Based Radar Odometry Designed For Scanning and Automotive Radars
"](https://arxiv.org/abs/2103.07908) , accepted for publication in the IEEE ICRA 2021.

### [Important] 

The result of this code is not same as the original paper becasue we change the optimization method in NDT to improve the robustness of scan matching. The PW-NDT can now be used without failure detection based on vehicle acceleration constrain. Therefore, the code we provided have removed complex rule for failure detection used in the origin paper.

## Radar Odometry on Oxford RobotCar Radar Dataset:

![](img/scanning_ro.gif)

Radar Odometry Result:

![](img/scanning_ro.gif)

# Getting Started
```

```

Normal Distribution Map Visualization:

![](img/normal_distribution_map.gif)
