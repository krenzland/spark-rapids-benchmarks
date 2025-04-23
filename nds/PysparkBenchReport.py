#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -----
#
# Certain portions of the contents of this file are derived from TPC-DS version 3.2.0
# (retrieved from www.tpc.org/tpc_documents_current_versions/current_specifications5.asp).
# Such portions are subject to copyrights held by Transaction Processing Performance Council (“TPC”)
# and licensed under the TPC EULA (a copy of which accompanies this file as “TPC EULA” and is also
# available at http://www.tpc.org/tpc_documents_current_versions/current_specifications5.asp) (the “TPC EULA”).
#
# You may not use this file except in compliance with the TPC EULA.
# DISCLAIMER: Portions of this file is derived from the TPC-DS Benchmark and as such any results
# obtained using this file are not comparable to published TPC-DS Benchmark results, as the results
# obtained from using this file do not comply with the TPC-DS Benchmark.
#

import json
import os
import time
import traceback
from typing import Callable
from pyspark.sql import SparkSession
import inspect

import python_listener

class PysparkBenchReport:
    """Class to generate json summary report for a benchmark
    """
    def __init__(self, spark_session: SparkSession, query_name) -> None:
        self.spark_session = spark_session
        self.summary = {
            'env': {
                'envVars': {},
                'sparkConf': {},
                'sparkVersion': None
            },
            'queryStatus': [],
            'exceptions': [],
            'startTime': None,
            'queryTimes': [],
            'query': query_name,
        }

    def report_on(self, fn: Callable, warmup_iterations = 0, iterations = 1, *args):
        """Record a function for its running environment, running status etc. and exclude sentive
        information like tokens, secret and password Generate summary in dict format for it.

        Args:
            fn (Callable): a function to be recorded. Can be either a regular function or a generator
                          function that yields after each sub-operation.
            warmup_iterations (int): number of warmup iterations
            iterations (int): number of actual iterations
            *args: arguments to pass to the function

        Returns:
            dict: summary of the fn
        """
        spark_conf = dict(self.spark_session.sparkContext._conf.getAll())
        env_vars = dict(os.environ)
        redacted = ["TOKEN", "SECRET", "PASSWORD"]
        filtered_env_vars = dict((k, env_vars[k]) for k in env_vars.keys() if not (k in redacted))
        self.summary['env']['envVars'] = filtered_env_vars
        self.summary['env']['sparkConf'] = spark_conf
        self.summary['env']['sparkVersion'] = self.spark_session.version
        listener = None
        try:
            listener = python_listener.PythonListener()
            listener.register()
        except TypeError as e:
            print("Not found com.nvidia.spark.rapids.listener.Manager", str(e))
            listener = None
        if listener is not None:
            print("TaskFailureListener is registered.")

        # Initialize sub-operation timing structure
        self.summary['subOperationTimes'] = []

        try:
            # warmup
            for i in range(0, warmup_iterations):
                if inspect.isgeneratorfunction(fn):
                    # For generator functions, we need to consume all yields
                    for _ in fn(*args):
                        pass
                else:
                    fn(*args)
        except Exception as e:
            print('ERROR WHILE WARMUP BEGIN')
            print(e)
            traceback.print_tb(e.__traceback__)
            print('ERROR WHILE WARMUP END')

        start_time = int(time.time() * 1000)
        self.summary['startTime'] = start_time
        
        # run the query
        for i in range(0, iterations):
            try:
                iteration_start = int(time.time() * 1000)
                if inspect.isgeneratorfunction(fn):
                    # Track sub-operations for generator functions
                    sub_ops = []
                    last_time = iteration_start
                    for op_name in fn(*args):
                        current_time = int(time.time() * 1000)
                        sub_ops.append({
                            'name': op_name,
                            'startTime': last_time,
                            'endTime': current_time,
                            'duration': current_time - last_time
                        })
                        last_time = current_time
                    end_time = last_time
                    self.summary['subOperationTimes'].append(sub_ops)
                else:
                    # Original behavior for regular functions
                    fn(*args)
                    end_time = int(time.time() * 1000)
                
                if listener and len(listener.failures) != 0:
                    self.summary['queryStatus'].append("CompletedWithTaskFailures")
                else:
                    self.summary['queryStatus'].append("Completed")
            except Exception as e:
                # print the exception to ease debugging
                print('ERROR BEGIN')
                print(e)
                traceback.print_tb(e.__traceback__)
                print('ERROR END')
                end_time = int(time.time() * 1000)
                self.summary['queryStatus'].append("Failed")
                self.summary['exceptions'].append(str(e))
            finally:
                self.summary['queryTimes'].append(end_time - iteration_start)
        if listener is not None:
            listener.unregister()
        return self.summary

    def write_summary(self, prefix=""):
        """_summary_

        Args:
            query_name (str): name of the query
            prefix (str, optional): prefix for the output json summary file. Defaults to "".
        """
        # Power BI side is retrieving some information from the summary file name, so keep this file
        # name format for pipeline compatibility
        filename = prefix + '-' + self.summary['query'] + '-' +str(self.summary['startTime']) + '.json'
        self.summary['filename'] = filename
        with open(filename, "w") as f:
            json.dump(self.summary, f, indent=2)

    def is_success(self):
        """Check if the query succeeded, queryStatus == Completed
        """
        return self.summary['queryStatus'][0] == 'Completed'
