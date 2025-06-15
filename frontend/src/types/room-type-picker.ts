import { RoomType } from "@/types/room-type.ts";

export const validRoomTypes = RoomType.entries().filter(
  ([_, value]) =>
    ![RoomType.FRONT_DOOR, RoomType.INTERIOR_DOOR, RoomType.UNKNOWN].includes(
      value,
    ),
);

export const defaultMap: Map<RoomType, number> = new Map(
  validRoomTypes.map(([_, value]) => [value, 0]),
);
