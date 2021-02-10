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
