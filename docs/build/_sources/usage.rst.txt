Usage
=====

.. _installation:

Installation
------------

To use sc_intNMF first install it using pip:

.. code-block:: console

    (.venv) $ pip install sc_intNMF

log TF-IDF
----------

To perform log term-frequency inverse document frequency (TF-IDF) transform of a single cell omics experiment (*i.e.* cell-feature matrix) you can use the ``nmf_models_mod_updates.log_tf_idf()`` functions:

.. autofunction:: nmf_models_mod_updates.log_tf_idf

The ``mat_in`` parameter should be either ``"scipy csr_matrix"``, ``"dense matrix or rank-2 ndarray D"``,
or ``"another sparse matrix"``. Otherwise, :py:func:`nmf_models_mod_updates.log_tf_idf`
will raise an exception.

.. autoexception:: nmf_models_mod_updates.InvalidMatType   
