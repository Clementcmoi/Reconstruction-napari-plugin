class LayerUtils:
    @staticmethod
    def connect_signals(widget):
        """
        Connect signals for layer updates.
        """
        widget.viewer.layers.events.inserted.connect(lambda event: LayerUtils.update_layer_selections(widget, event))
        widget.viewer.layers.events.removed.connect(lambda event: LayerUtils.update_layer_selections(widget, event))
        widget.viewer.layers.events.changed.connect(lambda event: LayerUtils.update_layer_selections(widget, event))
        widget.viewer.layers.events.reordered.connect(lambda event: LayerUtils.update_layer_selections(widget, event))

    @staticmethod
    def update_layer_selections(widget, event=None):
        """
        Update the QComboBox selections with the list of layers.
        Preserves the current sample selection across refreshes.
        If the selected layer was removed, falls back to the latest
        non-reconstruction layer.
        """
        layers = [layer.name for layer in widget.viewer.layers]

        # ── sample_selection ──────────────────────────────────────────────
        current = widget.sample_selection.currentText()
        widget.sample_selection.blockSignals(True)
        widget.sample_selection.clear()
        widget.sample_selection.addItems(layers)
        widget.sample_selection.blockSignals(False)

        # Restore previous selection if it still exists
        idx = widget.sample_selection.findText(current)
        if idx >= 0:
            widget.sample_selection.setCurrentIndex(idx)
        else:
            # Fall back to the latest layer that has no 'reconstruction' key in metadata
            for layer in reversed(widget.viewer.layers):
                if 'reconstruction' not in getattr(layer, 'metadata', {}):
                    fallback_idx = widget.sample_selection.findText(layer.name)
                    if fallback_idx >= 0:
                        widget.sample_selection.setCurrentIndex(fallback_idx)
                    break

        if hasattr(widget, 'darkfield_checkbox') and widget.darkfield_checkbox.isChecked() and widget.darkfield_selection:
            widget.darkfield_selection.clear()
            widget.darkfield_selection.addItems(layers)

        if hasattr(widget, 'flatfield_checkbox') and widget.flatfield_checkbox.isChecked() and widget.flatfield_selection:
            widget.flatfield_selection.clear()
            widget.flatfield_selection.addItems(layers)
