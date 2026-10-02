napari-dextrusion
=================

A napari plugin for `DeXtrusion <https://github.com/alvillars/dextrusion-torch>`_ (cellular events in
2D+t epithelial movies): open a movie, detect events, annotate, train, and review a sample of the
predicted events to retrain the network on its own corrected predictions.

It is written on top of ``dextrusion-torch`` and runs its command line (``detect``, ``prepare``,
``train``) as child processes, so the napari window never blocks.

Install
-------

.. code-block:: bash

    git clone https://github.com/alvillars/napari-dextrusion
    cd napari-dextrusion
    uv sync --extra gui
    uv run napari            # Plugins > DeXtrusion

Project folder
--------------

Opening ``movie.tif`` (a 2D projected ``T x Y x X`` movie) creates ``movie_dextrusion/`` next to it:
the movie is symlinked there and everything the plugin writes goes there. Coordinates are always those
of the movie you opened; the plugin rescales a training copy to the scale of the networks.

.. code-block:: text

    movie_dextrusion/
      movie.tif -> ...                symlink, never a copy
      movie_cell_<class>.zip          annotations / reviewed events (ImageJ point ROIs)
      movie_nothing.zip               hand-picked non-events (class 0)
      detect/<model>/                 probability maps, ROI files, plugin_detect.json (settings)
      prepared/                       rescaled training copy (made by Train)
      runs/<name>/                    trained model, history.csv
      movie_review_<model>.json       review verdicts (resumable)

Set the **cell diameter** and **event duration** of your movie at the top of the widget: they decide the
zoom applied to detection, training and to the review crop.

Tabs
----

Detect
    Pick a DeXNet folder (one net, or a folder of nets used as an ensemble) and run detection. Results
    appear as one Points layer per class. Settings are the ones of ``dextrusion detect``.

Annotate
    The annotation tool of ``dextrusion label``: one Points layer per class, keys ``q w g h j`` select
    the class, autosave with ``.bak.zip`` backups, resumable.

Train
    Rescales the project movie and its annotations (``dextrusion prepare``), then trains or fine-tunes
    (``dextrusion train``) and plots loss and accuracy live. Fine-tuning widens the output layer for
    new classes (``--init-from`` + ``--catnames``); *other data* links a second dataset (for instance
    the original one) next to the project movie, and *project movie x* oversamples a small movie.
    Defaults (10 epochs, lr 0.01, frozen CNN is opt-in) are a starting point, not a recommendation.
    Hand-picked non-events are used only when *non-event copies* is above 1.

Review
    Loads the detections of a Detect run, draws a random percentage of the events of each class and
    shows them one at a time as the exact window the network sees (10 frames, 45 x 45 px at the
    network's scale, mapped back to your movie), looping. Verdict keys: ``q w g h j`` one class each
    (in the order of the buttons), ``n`` not an event, ``u`` exclude. Verdicts are written at once:
    a class goes to that class's file, *not an event* to ``movie_nothing.zip``, *exclude* nowhere.
    Changing a verdict moves the point; points you annotated by hand are never removed. *Retrain*
    jumps to the Train tab.

Development
-----------

.. code-block:: bash

    uv sync --extra gui
    uv run ruff check src tests
    QT_QPA_PLATFORM=offscreen uv run pytest -q

The GUI-free logic is in ``review.py``, ``review_run.py``, ``commands.py`` and ``session.py``; the tabs
are thin. ``tests/test_gui.py`` runs the whole detect, annotate, fine-tune chain through the real
command line on a tiny synthetic movie.
