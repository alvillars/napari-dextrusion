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
    Loads the detections of a Detect run (listed in the drop-down) or of any earlier run: *Other
    detection folder...* takes a folder with ``<movie>_cell_*.zip`` ROI files. Maps
    (``*_rawproba.tif``) are optional and only give a score; pick the DeXNet used in *model* so the
    crop has its window size (empty: standard window). Then it draws a random percentage of the events of each class and
    shows them one at a time as the exact window the network sees (10 frames, 45 x 45 px at the
    network's scale, mapped back to your movie), looping. Verdict keys: ``q w g h j`` one class each
    (in the order of the buttons), ``n`` not an event, ``u`` exclude. Verdicts are written at once:
    a class goes to that class's file, *not an event* to ``movie_nothing.zip``, *exclude* nowhere.
    *Stop review* leaves at any time (the loop also pauses when you open another tab); *Start / resume*
    continues where you stopped. Changing a verdict moves the point; points you annotated by hand are never removed. *Retrain*
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

Tutorial
--------

This walks through a complete cycle on your own movie: detect events with an existing network,
correct what it found, and retrain it on the corrected data. Every step can also be run alone.

What you need
~~~~~~~~~~~~~

- A 2D movie as a ``.tif`` shaped ``T x Y x X`` (project 3D / multichannel data to 2D first).
- A DeXNet: a folder with ``model.safetensors`` + ``config.json`` (the ``models/`` folder of
  ``dextrusion-torch`` has several; ``dextrusion convert`` turns legacy Keras networks into this
  format). A folder holding several DeXNets is used as an ensemble.
- Ideally a GPU; the CPU works but detection and training are much slower.

1. Open the movie and set its scale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start napari (``uv run napari``) and open **Plugins > DeXtrusion**. Click *Open movie* (or select an
image layer that was read from a ``.tif`` and click *Use the selected image layer*).

The plugin creates ``<movie name>_dextrusion/`` next to the movie, links the movie into it (never a
copy) and keeps everything it writes there. Then set, at the top of the widget:

- **cell diameter (px)**: the typical cell diameter in *your* movie, in pixels. Measure a few cells
  in napari.
- **event duration (frames)**: how long an event lasts in your movie.

The networks were trained with cells of about 25 px and events of about 4.5 frames. If your values
differ by more than 30 %, the movie is rescaled for detection and training, and the review crop is
scaled the same way. These two numbers change the results, so measure them rather than guess.

2. Detect events
~~~~~~~~~~~~~~~~

In the **Detect** tab choose the *model(s)* folder, the device (``auto`` uses the GPU if there is
one) and, if needed, the window step and thresholds (the defaults are those of ``dextrusion detect``).
Click *Detect events*. The log shows the progress; *Stop* kills the run.

Results go to ``detect/<model name>/`` (a new folder for every run, nothing is overwritten): one
probability map and one ROI file per class, plus ``plugin_detect.json`` with the settings. One Points
layer per class is added to the viewer, named ``detected <class> (<run>)``.

If too much or too little is detected, raise or lower the *volume* and *probability* thresholds and
run again. They only affect which volumes of the probability maps become events.

3. Annotate
~~~~~~~~~~~

The **Annotate** tab is where you create your own training examples. Type the class names separated
by commas (for example ``division, delamination``; each class is saved as
``<movie>_cell_<name>.zip``, up to five classes) and click *Start annotating*.

- ``q w g h j`` select the class in the order of the buttons and switch to add mode; click on the
  cell to add a point at the *current frame*.
- Select points and press Backspace to delete them.
- Everything is saved automatically; the previous version of a file is kept as ``.bak.zip``. ``k``
  saves immediately. Existing files are loaded, so you can stop and continue.

Mark every event on the same landmark, at the frame and cell centre where it is most recognisable. A
training window covers 5 frames before to 4 frames after the marked frame and its position is
jittered by about 2 frames, so be consistent rather than exact.

4. Train or fine-tune
~~~~~~~~~~~~~~~~~~~~~

Open the **Train** tab.

- **start from**: a DeXNet to fine-tune, or empty to train from scratch.
- **classes**: when fine-tuning, the classes to add to the network's (names it already has are not
  added twice); from scratch, all the classes. These are the names used for the annotation files.
- **other data**: optional folder with other movies and ROI files (for example the original dataset),
  linked next to your movie so the network does not forget what it knew. Your own movie's files in
  that folder are ignored: the project folder always wins.
- **project movie x**: how many times your movie is sampled. A small annotated movie needs a factor
  above 1 next to a large dataset.
- **epochs / lr**, **augmentation**, **validation share / batch**: the usual training settings. If
  the log says that there are fewer training windows than one batch, lower the batch size or
  annotate more.
- **non-event copies**: copies of every hand-picked non-event (see Review). It must be above 1 or
  ``<movie>_nothing.zip`` is ignored.
- **freeze the CNN**: for small datasets, train only the recurrent part and the decision layers.

*Rescale + train* first rescales your movie and its ROI files to the scale of the networks
(``prepared/``), then trains. Loss and accuracy are drawn for the training and the validation set as
the epochs finish. The model is saved in ``runs/<run name>/``; when it is done, it is selected as the
model of the Detect tab and as the starting point of the next training.

The defaults (10 epochs, learning rate 0.01, 3x augmentation, 10 copies of every non-event) are a
starting point for fine-tuning on a few dozen events, not a recommendation. Watch the validation
curve; it only reflects the few events you annotated.

5. Review predictions and retrain on them
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Annotating every event by hand is slow. The **Review** tab lets you correct a sample of what the
network predicted, and turns the corrections into new training data.

1. Pick the detections in the drop-down (the runs made with the Detect tab are listed) and click
   *Load detections*. The number of events found per class is shown.
2. Set the percentage of the events of each class to review (a random sample, reproducible with the
   *seed*; at least one event per class if the percentage is above 0) and click
   *Start / resume review*.
3. Each event is shown as a looping crop of the video: exactly the window the network sees (10
   frames, 45 x 45 px at the network's scale, mapped back to your movie), with a ring on the event.
   Decide what it is:

   - ``q w g h j``: one class each, in the order of the buttons (also the buttons themselves);
   - ``n``: *not an event* (a false positive: it will be used as a hard negative);
   - ``u``: *exclude* (you cannot tell, or the crop is unusable: nothing is written).

   *Back* and *Skip* move through the queue. *Stop review* leaves at any time: every verdict is
   already saved, the movie is shown again, and *Start / resume review* continues where you stopped.
4. Verdicts are written as you go: a class goes to ``<movie>_cell_<class>.zip``, *not an event* to
   ``<movie>_nothing.zip``. Changing a verdict moves the point. Points you annotated by hand are never
   removed. The Annotate layers, if open, are refreshed.
5. Click *Retrain with these annotations* (or open the Train tab), check the settings and train.
   Then detect again with the new model and review a new sample: this is the loop.

Reviewing is also an estimate of the quality of the network: with a random sample, the share of
events you confirm per class is a fair estimate of its precision. It says nothing about the events
the network missed.

Detections from an earlier run
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Detections made outside the plugin (for example with ``dextrusion detect`` on the command line) can
be reviewed as well: click *Other detection folder...* and pick a folder with
``<movie>_cell_*.zip`` ROI files. The probability maps (``*_rawproba.tif``) are optional and only
give a score. Pick the DeXNet that made them in *model* so that the crop has its window size;
if you leave it empty the standard window is used. Detections that fall outside the open movie (made
on another movie) are refused.

Things to know
~~~~~~~~~~~~~~

- **Copy existing annotations first.** To continue from annotations you already have, copy their
  ``<movie>_cell_<class>.zip`` files into the project folder *before* reviewing: review adds to these
  files, whereas copying later overwrites them.
- **Never copy detected ROI files into the project folder.** They have the same names as the
  annotation files and would be trained on as if they were your annotations.
- **A confirmed prediction on an already annotated cell is added as a second point** a few pixels from
  the first: that event then counts twice in training. Keep an eye on this when you review detections
  of a network that was trained on your annotations.
- **Keep the project movie's data in the project folder.** *Rescale + train* writes the rescaled movie
  and ROI files to ``prepared/``; it refuses to start when ``prepared/`` holds symlinks named like your
  movie.
- **Leave the cell diameter and duration as they were** between detection, review and training of the
  same project, or the crop and the rescaling no longer match the detections.
- **Large movies** take a while: detection and training run as separate processes whose log is shown
  in the tab, and they can be stopped.

What the plugin writes
~~~~~~~~~~~~~~~~~~~~~~

========================================  =====================================================
``<movie>_cell_<class>.zip``              annotated and reviewed events (movie coordinates)
``<movie>_nothing.zip``                   hand-picked non-events (class 0)
``<movie>_review_<run>.json``             review queue and verdicts of one detection run
``detect/<model>/``                       probability maps, ROI files, ``plugin_detect.json``
``prepared/``                             rescaled training copy, written by Train
``runs/<name>/``                          trained model and ``history.csv``
========================================  =====================================================

The ROI files are standard ImageJ point ROIs, so they can be used with ``dextrusion train`` and
``dextrusion evaluate`` from the command line as well.
