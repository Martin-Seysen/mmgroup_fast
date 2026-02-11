

===========================================
The C interface of the mmgroup_fast project
===========================================


Introduction
============

This *document* describes the functionality of the C modules in this
project. For most of these C modules there is also a python extension.
Unless otherwise stated, each documented C function is wrapped by a
Cython function with the same name and signature. 

Note that almost all parameters of such a C function are declared as
(signed or unsigend) integers, or as pointers to such integers. In
python, a ``numpy`` array of appropriate ``dtype`` may be passed as
an argument to a parameter delared as a pointer to an integer.
  


.. highlight:: none


Description of the ``mmgroup_fast`` extension
=============================================

This section is yet to be written


C interface
-----------

Header files
............

.. doxygenfile:: mm_op_fast.h


C functions dealiong with matrices in the Griess algebra
.........................................................

.. doxygenfile:: mm_op_fast_word.c



Shared libraries and dependencies between Cython extensions
===========================================================

This is yet to be written


