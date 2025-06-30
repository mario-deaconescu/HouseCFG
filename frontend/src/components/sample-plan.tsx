import { useCallback, useEffect, useMemo, useState } from "react";
import { Tooltip } from "@heroui/tooltip";

import { PlanModel, RoomModel } from "@/types/sample.ts";
import { RoomType } from "@/types/room-type.ts";

type Props = {
  plan: PlanModel;
  scaleFactor?: number;
};

const canvasSize = 256;

const colorMap = new RoomType.ColorMap("rgbaString", 1);

function polygonCentroid(points: [number, number][]) {
  const n = points.length;

  if (n < 3) throw new Error("Need at least 3 points for a polygon");

  let area = 0,
    cx = 0,
    cy = 0;

  for (let i = 0; i < n; i++) {
    const [x0, y0] = points[i];
    const [x1, y1] = points[(i + 1) % n]; // Wrap around

    const a = x0 * y1 - x1 * y0;

    area += a;
    cx += (x0 + x1) * a;
    cy += (y0 + y1) * a;
  }

  area *= 0.5;
  cx /= 6 * area;
  cy /= 6 * area;

  return {
    centroid: [cx, cy],
    area: Math.abs(area),
  };
}

function pointInPolygon(point: [number, number], polygon: [number, number][]) {
  const [px, py] = point;
  let inside = false;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];

    const intersect =
      yi > py !== yj > py &&
      px < ((xj - xi) * (py - yi)) / (yj - yi + 1e-10) + xi;

    if (intersect) inside = !inside;
  }

  return inside;
}

const getRoomTypeName = (room: RoomModel) => {
  // @ts-ignore
  const key = RoomType.keys().find((key) => key == room.room_type);

  return key === undefined ? "" : RoomType.keyToString(RoomType[key]);
};

const SamplePlan = ({ plan, scaleFactor = 1 }: Props) => {
  const [canvasRef, setCanvasRef] = useState<HTMLCanvasElement | null>(null);

  const validRooms = useMemo(() => {
    return plan.rooms.filter((room) => {
      const area = polygonCentroid(room.corners).area;

      return area >= 10;
    });
  }, [plan]);

  const roomData = useMemo(() => {
    const data = validRooms.map((room) => {
      return polygonCentroid(room.corners);
    });

    console.log(data.map(({ area }) => area));
    console.log(data.filter(({ area }) => area >= 10));

    return data;
  }, [validRooms]);

  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  useEffect(() => {
    if (!canvasRef) return;
    const ctx = canvasRef.getContext("2d");

    if (!ctx) return;
    ctx.clearRect(0, 0, canvasRef.width, canvasRef.height);

    for (const room of validRooms) {
      ctx.fillStyle = colorMap.getColor(room.room_type);
      ctx.beginPath();
      ctx.moveTo(
        room.corners[0][0] * scaleFactor,
        room.corners[0][1] * scaleFactor,
      );
      for (let i = 1; i < room.corners.length; i++) {
        ctx.lineTo(
          room.corners[i][0] * scaleFactor,
          room.corners[i][1] * scaleFactor,
        );
      }
      ctx.closePath();
      ctx.fill();
    }

    ctx.fillStyle = "black";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (const [index, room] of validRooms.entries()) {
      const { centroid, area } = roomData[index];

      ctx.font = `${Math.floor(0.2 * scaleFactor * Math.sqrt(area))}px Arial`;

      const text = getRoomTypeName(room);

      ctx.fillText(text, centroid[0] * scaleFactor, centroid[1] * scaleFactor);
    }
  }, [canvasRef, scaleFactor, validRooms]);

  const canvasRect = useMemo(() => {
    return canvasRef?.getBoundingClientRect();
  }, [canvasRef]);

  const onMouseMove = useCallback(
    (event: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
      if (!canvasRect) return;
      // Transform to canvas coordinates
      const [globalX, globalY] = [event.clientX, event.clientY];
      const [canvasX, canvasY] = [
        globalX - canvasRect.left,
        globalY - canvasRect.top,
      ];

      // Transform to room coordinates
      const roomX = (canvasX / canvasRect.width) * canvasSize;
      const roomY = (canvasY / canvasRect.height) * canvasSize;

      // Find the room that contains the point
      const roomIndex = validRooms.findIndex((room) => {
        return pointInPolygon([roomX, roomY], room.corners);
      });

      setHoveredIndex(roomIndex === -1 ? null : roomIndex);
    },
    [canvasRect, setHoveredIndex],
  );

  const onMouseLeave = useCallback(() => {
    setHoveredIndex(null);
  }, [setHoveredIndex]);

  return (
    <div
      className={"w-full h-full relative"}
      onMouseLeave={onMouseLeave}
      onMouseMove={onMouseMove}
    >
      <canvas
        ref={setCanvasRef}
        className={"w-full h-full"}
        height={canvasSize * scaleFactor}
        width={canvasSize * scaleFactor}
      />
      {canvasRect && (
        <>
          {validRooms.map((room, index) => (
            <Tooltip
              key={index}
              classNames={{
                base: "pointer-events-none",
                content: "pointer-events-none",
              }}
              content={getRoomTypeName(room)}
              isOpen={hoveredIndex === index}
              size={"lg"}
            >
              <div
                className={"w-0 h-0 bg-amber-200 absolute"}
                style={{
                  left:
                    (roomData[index].centroid[0] / canvasSize) *
                    canvasRect.width,
                  top:
                    (roomData[index].centroid[1] / canvasSize) *
                      canvasRect.height +
                    15,
                }}
              />
            </Tooltip>
          ))}
        </>
      )}
    </div>
  );
};

export default SamplePlan;
