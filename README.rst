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

C: Conv layer same as in original architecture

L: Lambda layer

Blank cell represents that the layer is removed (replaced with identity layer)

+----+----------------------------------------------------------------+------------+------------+
|    |                  Layer number and Type                         |            |            |
+----+---+---+---+---+---+---+---+---+---+---+----+----+----+----+----+------------+------------+
| Id | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | Params (M) |  Top-1 (%) |
+----+---+---+---+---+---+---+---+---+---+---+----+----+----+----+----+------------+------------+
| A1 | C | C | C | C | C | C | C | C | C | C | C  |  C |  C |  C | FC |    3.30    |    65.54   |
+----+---+---+---+---+---+---+---+---+---+---+----+----+----+----+----+------------+------------+
| A2 | C | C | C | C | C | C | C | C |   |   |    |    |  L |  C | FC |    1.84    |    69.48   |
+----+---+---+---+---+---+---+---+---+---+---+----+----+----+----+----+------------+------------+
| A3 | C | C | C | C | C | C | C | C | L | C | L  | C  |  L |  C | FC |    2.53    |    65.25   |
+----+---+---+---+---+---+---+---+---+---+---+----+----+----+----+----+------------+------------+
| A4 | C | C | C | C | C | C | C | C | L |   |    |    |  C |    | FC |    1.24    |    68.22   |
+----+---+---+---+---+---+---+---+---+---+---+----+----+----+----+----+------------+------------+
| A5 | C | C | C | C | C | C | L |   |   |   |    |    |  C |    | FC |    0.80    |    69.91   |
+----+---+---+---+---+---+---+---+---+---+---+----+----+----+----+----+------------+------------+
| A6 | C | C | C |   | C |   | L |   |   |   |    |    |  C |    | FC |    0.71    |    66.38   |
+----+---+---+---+---+---+---+---+---+---+---+----+----+----+----+----+------------+------------+

The above table shows significant gain in A5 configuration as compared to original configuration A1.

Citations
=========

.. code-block:: cite1

      @inproceedings{
          anonymous2021lambdanetworks,
          title={LambdaNetworks: Modeling long-range Interactions without Attention},
          author={Anonymous},
          booktitle={Submitted to International Conference on Learning Representations},
          year={2021},
          url={https://openreview.net/forum?id=xTJEN-ggl1b},
          note={under review}
      }

.. code-block:: cite1

      @article{DBLP:journals/corr/HowardZCKWWAA17,
           author    = {Andrew G. Howard and
                        Menglong Zhu and
                        Bo Chen and
                        Dmitry Kalenichenko and
                        Weijun Wang and
                        Tobias Weyand and
                        Marco Andreetto and
                        Hartwig Adam},
           title     = {MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
                        Applications},
           journal   = {CoRR},
           volume    = {abs/1704.04861},
           year      = {2017},
           url       = {http://arxiv.org/abs/1704.04861},
           archivePrefix = {arXiv},
           eprint    = {1704.04861},
           timestamp = {Mon, 13 Aug 2018 16:46:35 +0200},
           biburl    = {https://dblp.org/rec/journals/corr/HowardZCKWWAA17.bib},
           bibsource = {dblp computer science bibliography, https://dblp.org}
         }


