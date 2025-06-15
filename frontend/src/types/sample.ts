import { RoomType } from "@/types/room-type.ts";

export interface RoomModel {
  room_type: RoomType;
  corners: [number, number][];
}

export interface PlanModel {
  rooms: RoomModel[];
}

export interface SampleList {
  final: boolean;
  images: number[][][][];
  original_images: number[][][][];
  plans: (PlanModel | null)[];
}

export interface SampleImage {
  image: number[][][];
}
