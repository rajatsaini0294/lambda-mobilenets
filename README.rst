=================
Lambda Mobilenets
=================

Mobilenets with Lambda layers

Lambda Networks proposed in `LambdaNetworks: Modeling Long Range Interactions without Attention <https://openreview.net/pdf?id=xTJEN-ggl1b>`_.

Install
=======

We use the implementation done by `lucidrains <https://github.com/lucidrains/lambda-networks>`_

.. code-block:: bash

   pip install lambda-networks
   
   
Method
======

We replace some layers in MobileNet-v1 architecture with lambda layer. It significantly reduces the parameters with some performance gain on cifar100 dataset.

MobileNet-v1 architecture:

.. image:: assets/mnetv1.svg
   :height: 300px
   :align: center

The following table shows which layer to replace and remove to get the performance boost:

+----+------------------------------------------------------------+------------+------------+
| Id |                  Layer Type                                |  Params (M)|  Top-1 (%) |
+----+---+---+---+---+---+---+---+---+---+---+---+---+---+---+----+------------+------------+
|  1 | C | C | C | C | C | C | C | C | C | C | C | C | C | C | FC |    3.30    |    65.54   |
+----+---+---+---+---+---+---+---+---+---+---+---+---+---+---+----+------------+------------+
|  2 | C | C | C | C | C | C | C | C |   |   |   |   | L | C | FC |    1.84    |    69.48   |
+----+---+---+---+---+---+---+---+---+---+---+---+---+---+---+----+------------+------------+
|  3 | C | C | C | C | C | C | C | C | L | C | L | C | L | C | FC |    2.53    |    65.25   |
+----+---+---+---+---+---+---+---+---+---+---+---+---+---+---+----+------------+------------+
|  4 | C | C | C | C | C | C | C | C | L |   |   |   | C |   | FC |    1.24    |    68.22   |
+----+---+---+---+---+---+---+---+---+---+---+---+---+---+---+----+------------+------------+
|  5 | C | C | C | C | C | C | L |   |   |   |   |   | C |   | FC |    0.80    |    69.91   |
+----+---+---+---+---+---+---+---+---+---+---+---+---+---+---+----+------------+------------+
|  6 | C | C | C |   | C |   | L |   |   |   |   |   | C |   | FC |    0.71    |    66.38   |
+----+---+---+---+---+---+---+---+---+---+---+---+---+---+---+----+------------+------------+
