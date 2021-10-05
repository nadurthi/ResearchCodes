#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 19:09:40 2021

@author: na0043
"""
import numpy
import g2o

class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False):
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(id)
        v_se3.set_estimate(pose)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_edge(self, vertices, measurement, 
            information=np.identity(6),
            robust_kernel=g2o.RobustKernelHuber()):

        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(measurement)  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(id).estimate()
    
    
pp=PoseGraphOptimization()
pp.add_vertex(0,g2o.Isometry3d(np.identity(3), [0, 0, 0]),fixed=True)
pp.add_vertex(1,g2o.Isometry3d(np.identity(3), [1, 1, 0]),fixed=False)
pp.add_vertex(2,g2o.Isometry3d(np.identity(3), [2, 2, 0]),fixed=False)

pp.add_edge([0,1],g2o.Isometry3d(np.identity(3), [0.9, 0.9, 0]))
pp.add_edge([1,2],g2o.Isometry3d(np.identity(3), [1.9, 1.9, 0]))

pp.optimize()

v1=pp.get_pose(0)
v1.position()
v1.rotation_matrix()

v2=pp.get_pose(1)
v2.position()
v2.rotation_matrix()


v3=pp.get_pose(2)
v3.position()
v3.rotation_matrix()
