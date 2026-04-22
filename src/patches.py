from __future__ import annotations


def _patch_tal_topk(topk: int) -> None:
    """
    Parchea TaskAlignedAssigner para asignar mas anchors positivos por box.
    Default Ultralytics = 10. Para RFIs elongados con bajo IoU de anchor
    assignment, subir a 13-15 da mas senal de gradiente a cada box dificil.
    Funciona en YOLOv8, YOLO11, yolo26 (cualquier modelo con TAL).
    """
    try:
        import inspect

        from ultralytics.utils.tal import TaskAlignedAssigner

        _orig = TaskAlignedAssigner.__init__
        default_topk = topk

        orig_signature = inspect.signature(_orig)

        def _patched(
            self,
            *args,
            topk=None,
            _topk=None,
            topk2=None,
            _topk2=None,
            num_classes=80,
            alpha=0.5,
            beta=6.0,
            eps=1e-9,
            **kwargs,
        ):
            chosen_topk = topk if topk is not None else _topk if _topk is not None else default_topk
            chosen_topk2 = topk2 if topk2 is not None else _topk2

            call_kwargs = dict(kwargs)
            if "topk" in orig_signature.parameters:
                call_kwargs["topk"] = chosen_topk
            elif "_topk" in orig_signature.parameters:
                call_kwargs["_topk"] = chosen_topk

            if chosen_topk2 is not None:
                if "topk2" in orig_signature.parameters:
                    call_kwargs["topk2"] = chosen_topk2
                elif "_topk2" in orig_signature.parameters:
                    call_kwargs["_topk2"] = chosen_topk2

            if "num_classes" in orig_signature.parameters:
                call_kwargs["num_classes"] = num_classes
            if "alpha" in orig_signature.parameters:
                call_kwargs["alpha"] = alpha
            if "beta" in orig_signature.parameters:
                call_kwargs["beta"] = beta
            if "eps" in orig_signature.parameters:
                call_kwargs["eps"] = eps

            _orig(self, *args, **call_kwargs)

        TaskAlignedAssigner.__init__ = _patched
        print(f"[tal] TaskAlignedAssigner topk -> {topk}")
    except Exception as e:
        print(f"[tal] No se pudo parchear TaskAlignedAssigner: {e}")


def _patch_channel_augmentations(enable: bool) -> None:
    """
    Activa augmentaciones de robustez por canal en Ultralytics:
      - ChannelShuffle
      - ChannelDropout

    Objetivo: reducir dependencia del modelo en un canal fijo para detectar RFI.
    """
    if not enable:
        return

    try:
        from ultralytics.data.augment import Albumentations as _Albumentations

        _orig_init = _Albumentations.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            if not getattr(self, "transform", None):
                return

            try:
                import albumentations as A

                self.transform.transforms.append(A.ChannelShuffle(p=0.30))
                self.transform.transforms.append(
                    A.ChannelDropout(
                        channel_drop_range=(1, 1),
                        fill=0,
                        p=0.20,
                    )
                )
                print("[aug] ChannelShuffle + ChannelDropout activados")
            except Exception as inner_e:
                print(f"[aug] No se pudieron agregar augmentaciones de canal: {inner_e}")

        _Albumentations.__init__ = _patched_init
    except Exception as e:
        print(f"[aug] No se pudo parchear Albumentations: {e}")
