import { useCallback, useEffect, useMemo, useState } from "react";

import { forEachInMask, Mask } from "@/types/mask.ts";
import Canvas from "@/components/canvas.tsx";
import { canvasSize } from "@/types/canvas.ts";
import { RoomType } from "@/types/room-type.ts";
import { Bubble } from "@/types/bubble-mask.ts";

const enum CursorState {
  DEFAULT,
  DRAW,
  INVALID,
  DELETE,
}

type Props = {
  className?: string;
  scaleFactor?: number;
  size?: number;
  selectedRoomType: RoomType;
  backgroundMask: Mask<boolean>;
  onChange?: (bubbles: Bubble[]) => void;
};

const colorMap = new RoomType.ColorMap("rgbaString", 0.5);
const hoverColorMap = new RoomType.ColorMap("rgbaString", 0.2);

const BubbleCanvas = ({
  className,
  scaleFactor = 1,
  size = 1,
  selectedRoomType,
  backgroundMask,
  onChange,
}: Props) => {
  const [bubbles, setBubbles] = useState<Bubble[]>([]);
  const [mousePosition, setMousePosition] = useState<[number, number] | null>(
    null,
  );

  const toCanvasRef = useCallback(
    (x: number, y: number): [number, number] => {
      return [Math.floor(x / scaleFactor), Math.floor(y / scaleFactor)];
    },
    [scaleFactor],
  );

  const tempBubble = useMemo(() => {
    if (mousePosition === null) {
      return null;
    }
    const [x, y] = mousePosition;
    const canvasMousePosition = toCanvasRef(x, y);

    return new Bubble(canvasMousePosition, size, selectedRoomType);
  }, [mousePosition, size, selectedRoomType, toCanvasRef]);

  const intersectionIndex = useMemo(() => {
    if (mousePosition === null || tempBubble === null) {
      return {
        pointer: null,
        bubble: null,
      };
    }
    const [x, y] = mousePosition;
    let bubbleIntersectionIndex: number | null = null;
    let intersectionIndex: number | null = null;

    const canvasMousePosition = toCanvasRef(x, y);

    for (const [i, bubble] of bubbles.entries()) {
      if (bubble.intersectsPoint(...canvasMousePosition)) {
        intersectionIndex = i;
      }
      if (bubble.intersectsBubble(tempBubble)) {
        bubbleIntersectionIndex = i;
      }
    }

    return {
      pointer: intersectionIndex,
      bubble: bubbleIntersectionIndex,
    };
  }, [mousePosition, size, selectedRoomType, bubbles, toCanvasRef, tempBubble]);

  const cursorState = useMemo(() => {
    const { pointer, bubble } = intersectionIndex;

    if (tempBubble === null) {
      return CursorState.DEFAULT;
    }
    if (pointer !== null) {
      return CursorState.DELETE;
    }
    if (bubble !== null) {
      return CursorState.INVALID;
    }

    return CursorState.DRAW;
  }, [tempBubble, intersectionIndex]);

  const cursorStyle = useMemo(() => {
    switch (cursorState) {
      case CursorState.DEFAULT:
        return "default";
      case CursorState.INVALID:
        return "not-allowed";
      case CursorState.DELETE:
        return "custom-/delete.cur";
      case CursorState.DRAW:
        return "crosshair";
    }
  }, [cursorState]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      ctx.clearRect(0, 0, canvasSize * scaleFactor, canvasSize * scaleFactor);
      ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
      forEachInMask((i, j) => {
        if (backgroundMask[i][j]) {
          ctx.fillRect(
            i * scaleFactor,
            j * scaleFactor,
            scaleFactor,
            scaleFactor,
          );
        }
      });

      const { pointer: pointerIndex, bubble: bubbleIndex } = intersectionIndex;

      for (const [i, bubble] of bubbles.entries()) {
        const map = i === pointerIndex ? hoverColorMap : colorMap;

        ctx.fillStyle = map.getColor(bubble.type);
        ctx.beginPath();
        ctx.arc(
          bubble.origin[0] * scaleFactor + 0.5 * scaleFactor,
          bubble.origin[1] * scaleFactor + 0.5 * scaleFactor,
          bubble.radius * scaleFactor,
          0,
          Math.PI * 2,
        );
        ctx.fill();
        ctx.fillStyle = "black";
        ctx.font = `${0.3 * scaleFactor * bubble.radius}px sans-serif`;
        const text = RoomType.keyToString(RoomType[bubble.type]);

        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(
          text,
          (bubble.origin[0] + 0.5) * scaleFactor,
          (bubble.origin[1] + 0.5) * scaleFactor,
        );
      }

      if (mousePosition && tempBubble) {
        ctx.fillStyle = hoverColorMap.getColor(selectedRoomType);

        let mode: "fill" | "stroke" = "fill";

        if (pointerIndex !== null) {
          ctx.fillStyle = "rgba(0,0,0,0)";
        } else if (bubbleIndex !== null || tempBubble.isOutOfBounds()) {
          ctx.strokeStyle = "rgba(255, 0, 0, 0.8)";
          mode = "stroke";
        } else {
          ctx.fillStyle = hoverColorMap.getColor(tempBubble.type);
        }
        ctx.beginPath();
        ctx.arc(
          tempBubble.origin[0] * scaleFactor + 0.5 * scaleFactor,
          tempBubble.origin[1] * scaleFactor + 0.5 * scaleFactor,
          tempBubble.radius * scaleFactor,
          0,
          Math.PI * 2,
        );
        mode === "fill" ? ctx.fill() : ctx.stroke();
      }
    },
    [size, backgroundMask, mousePosition, selectedRoomType, tempBubble],
  );

  const onMouseMove = useCallback(
    (x: number, y: number) => {
      if (
        x < 0 ||
        x >= canvasSize * scaleFactor ||
        y < 0 ||
        y >= canvasSize * scaleFactor
      ) {
        setMousePosition(null);
      }
      setMousePosition([x, y]);
    },
    [setMousePosition, scaleFactor],
  );

  const onClick = useCallback(() => {
    const { pointer, bubble } = intersectionIndex;

    if (tempBubble === null) {
      return;
    }

    if (bubble === null && !tempBubble.isOutOfBounds()) {
      setBubbles((prevBubbles) => [...prevBubbles, tempBubble]);
    } else if (pointer !== null) {
      // Remove the bubble if it already exists
      setBubbles((prevBubbles) => prevBubbles.filter((_, i) => i !== pointer));
    }
  }, [bubbles, scaleFactor, size, selectedRoomType, onChange, tempBubble]);

  const onMouseLeave = useCallback(() => {
    setMousePosition(null);
  }, [setMousePosition]);

  useEffect(() => {
    onChange?.(bubbles);
  }, [bubbles, onChange]);

  return (
    <Canvas
      className={className}
      cursor={cursorStyle}
      draw={draw}
      height={canvasSize * scaleFactor}
      width={canvasSize * scaleFactor}
      onClick={onClick}
      onMouseLeave={onMouseLeave}
      onMouseMove={onMouseMove}
    />
  );
};

export default BubbleCanvas;
