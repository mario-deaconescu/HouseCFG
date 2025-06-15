export interface ModelParameters {
  condition_scale: number;
  rescaled_phi: number;
  ddim: boolean;
  skeletonize: boolean;
  simplify: boolean;
  felzenszwalb: boolean;
}

export const defaultModelParameters: ModelParameters = {
  condition_scale: 1,
  rescaled_phi: 0,
  ddim: true,
  skeletonize: true,
  simplify: true,
  felzenszwalb: true,
};
