###############################################################################
#   ilastik: interactive learning and segmentation toolkit
#
#       Copyright (C) 2011-2014, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# In addition, as a special exception, the copyright holders of
# ilastik give you permission to combine ilastik with applets,
# workflows and plugins which are not covered under the GNU
# General Public License.
#
# See the LICENSE file for details. License information is also available
# on the ilastik web site at:
#		   http://ilastik.org/license.html
###############################################################################

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Oct 14, 2014 16:31:56 EDT$"



from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.operators.opBlockedArrayCache import OpBlockedArrayCache

from ilastik.applets.base.applet import DatasetConstraintError

import itertools

import numpy

import nanshe
import nanshe.imp.segment
import nanshe.util.xnumpy


class OpNansheRemoveZeroedLines(Operator):
    """
    Given an input image and max/min bounds,
    masks out (i.e. sets to zero) all pixels that fall outside the bounds.
    """
    name = "OpNansheRemoveZeroedLines"
    category = "Pointwise"

    Input = InputSlot(allow_mask=True)

    ErosionShape = InputSlot(value=[21, 1])
    DilationShape = InputSlot(value=[1, 3])

    Output = OutputSlot(allow_mask=True)

    def __init__(self, *args, **kwargs):
        super( OpNansheRemoveZeroedLines, self ).__init__( *args, **kwargs )

        self._generation = {self.name : 0}

        self.Input.notifyReady( self._checkConstraints )

    def _checkConstraints(self, *args):
        slot = self.Input

        sh = slot.meta.shape
        ax = slot.meta.axistags
        if (len(slot.meta.shape) != 4) and (len(slot.meta.shape) != 5):
            # Raise a regular exception.  This error is for developers, not users.
            raise RuntimeError("was expecting a 4D or 5D dataset, got shape=%r" % (sh,))

        if "t" not in slot.meta.getTaggedShape():
            raise DatasetConstraintError(
                "RemoveZeroedLines",
                "Input must have time.")

        if "y" not in slot.meta.getTaggedShape():
            raise DatasetConstraintError(
                "RemoveZeroedLines",
                "Input must have space dim y.")

        if "x" not in slot.meta.getTaggedShape():
            raise DatasetConstraintError(
                "RemoveZeroedLines",
                "Input must have space dim x.")

        if "c" not in slot.meta.getTaggedShape():
            raise DatasetConstraintError(
                "RemoveZeroedLines",
                "Input must have channel.")

        numChannels = slot.meta.getTaggedShape()["c"]
        if numChannels != 1:
            raise DatasetConstraintError(
                "RemoveZeroedLines",
                "Input image must have exactly one channel.  " +
                "You attempted to add a dataset with {} channels".format( numChannels ) )

        if not ax[0].isTemporal():
            raise DatasetConstraintError(
                "RemoveZeroedLines",
                "Input image must have time first." )

        if not ax[-1].isChannel():
            raise DatasetConstraintError(
                "RemoveZeroedLines",
                "Input image must have channel last." )

        for i in range(1, len(ax) - 1):
            if not ax[i].isSpatial():
                # This is for developers.  Don't need a user-friendly error.
                raise RuntimeError("%d-th axis %r is not spatial" % (i, ax[i]))

    def setupOutputs(self):
        # Copy the input metadata to both outputs
        self.Output.meta.assignFrom( self.Input.meta )
        self.Output.meta.dtype = numpy.float32

        self.Output.meta.generation = self._generation

    def execute(self, slot, subindex, roi, result):
        key = roi.toSlice()
        raw = self.Input[key].wait()
        raw = raw[..., 0]
        raw = numpy.asanyarray(raw, numpy.float32)

        erosion_shape = self.ErosionShape.value
        dilation_shape = self.DilationShape.value

        processed = raw.copy()
        processed.fill(0)

        processed[:, ~numpy.ma.getmaskarray(raw).max(axis=0)] = nanshe.imp.segment.remove_zeroed_lines(
            nanshe.util.xnumpy.truncate_masked_frames(raw),
            erosion_shape=erosion_shape,
            dilation_shape=dilation_shape
        ).reshape(len(raw), -1)

        processed = processed[..., None]

        if slot.name == 'Output':
            result[...] = processed

    def setInSlot(self, slot, subindex, roi, value):
        pass

    def propagateDirty(self, slot, subindex, roi):
        if (slot.name == "Input") or (slot.name == "ErosionShape") or (slot.name == "DilationShape"):
            self._generation[self.name] += 1
            self.Output.setDirty( slice(None) )
        else:
            assert False, "Unknown dirty input slot"


class OpNansheRemoveZeroedLinesCached(Operator):
    """
    Given an input image and max/min bounds,
    masks out (i.e. sets to zero) all pixels that fall outside the bounds.
    """
    name = "OpNansheRemoveZeroedLinesCached"
    category = "Pointwise"


    Input = InputSlot(allow_mask=True)

    ErosionShape = InputSlot(value=[21, 1])
    DilationShape = InputSlot(value=[1, 3])

    Output = OutputSlot(allow_mask=True)

    def __init__(self, *args, **kwargs):
        super( OpNansheRemoveZeroedLinesCached, self ).__init__( *args, **kwargs )

        self.opRemoveZeroedLines = OpNansheRemoveZeroedLines(parent=self)

        self.opRemoveZeroedLines.ErosionShape.connect(self.ErosionShape)
        self.opRemoveZeroedLines.DilationShape.connect(self.DilationShape)


        self.opCache = OpBlockedArrayCache(parent=self)
        self.opCache.fixAtCurrent.setValue(False)

        self.opRemoveZeroedLines.Input.connect( self.Input )
        self.opCache.Input.connect( self.opRemoveZeroedLines.Output )
        self.Output.connect( self.opCache.Output )

    def setupOutputs(self):
        axes_shape_iter = itertools.izip(self.opRemoveZeroedLines.Output.meta.axistags,
                                         self.opRemoveZeroedLines.Output.meta.shape)

        block_shape = []

        for each_axistag, each_len in axes_shape_iter:
            if each_axistag.isSpatial():
                each_len = 256
            elif each_axistag.isTemporal():
                each_len = 50

            block_shape.append(each_len)

        block_shape = tuple(block_shape)

        self.opCache.innerBlockShape.setValue(block_shape)
        self.opCache.outerBlockShape.setValue(block_shape)

    def setInSlot(self, slot, subindex, roi, value):
        pass

    def propagateDirty(self, slot, subindex, roi):
        pass
