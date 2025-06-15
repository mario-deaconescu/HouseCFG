import { useEffect, useState } from "react";
import { Popover, PopoverContent, PopoverTrigger } from "@heroui/popover";
import { Button } from "@heroui/button";
import { Slider } from "@heroui/slider";
import { Checkbox } from "@heroui/checkbox";

import {
  defaultModelParameters,
  ModelParameters,
} from "@/types/model-parameters.ts";

type ParameterProps = {
  values?: ModelParameters;
  onChange?: (parameters: ModelParameters) => void;
};

const ModelParameterPicker = ({
  onChange,
  values = defaultModelParameters,
}: ParameterProps) => {
  const [conditionScale, setConditionScale] = useState(values.condition_scale);
  const [rescaledPhi, setRescaledPhi] = useState(values.rescaled_phi);
  const [ddim, setDdim] = useState(values.ddim);
  const [skeletonize, setSkeletonize] = useState(values.skeletonize);
  const [simplify, setSimplify] = useState(values.skeletonize);
  const [felzenszwalb, setFelzenszwalb] = useState(values.felzenszwalb);

  useEffect(() => {
    onChange?.({
      condition_scale: conditionScale,
      rescaled_phi: rescaledPhi,
      ddim: ddim,
      skeletonize: skeletonize,
      simplify: simplify,
      felzenszwalb: felzenszwalb,
    });
  }, [conditionScale, rescaledPhi, ddim, skeletonize, simplify, felzenszwalb]);

  return (
    <Popover placement={"top"}>
      <PopoverTrigger>
        <Button>Advanced</Button>
      </PopoverTrigger>
      <PopoverContent>
        {(titleProps) => (
          <>
            <div className={"px-5 py-5 w-full flex flex-col gap-5"}>
              <p className={"text-xl font-bold mb-2"} {...titleProps}>
                Model Parameters
              </p>
              <Slider
                className="w-full flex-grow"
                defaultValue={0}
                // endContent={<p className={"text-nowrap"}>Condition Scale</p>}
                label={"Rescaled Phi"}
                maxValue={1}
                minValue={0}
                showTooltip={true}
                step={0.1}
                value={rescaledPhi}
                onChange={(value) =>
                  typeof value == "number" && setRescaledPhi(value)
                }
              />
              <Slider
                className="w-full flex-grow"
                defaultValue={1}
                // endContent={<p className={"text-nowrap"}>Condition Scale</p>}
                label={"Condition Scale"}
                maxValue={3}
                minValue={0}
                showTooltip={true}
                step={0.1}
                value={conditionScale}
                onChange={(value) =>
                  typeof value == "number" && setConditionScale(value)
                }
              />

              <Checkbox
                aria-label={"Thin Walls"}
                isSelected={skeletonize}
                onValueChange={setSkeletonize}
              >
                Thin Walls
              </Checkbox>
              <Checkbox
                aria-label={"Simplify"}
                isSelected={simplify}
                onValueChange={setSimplify}
              >
                Simplify
              </Checkbox>
              <Checkbox
                aria-label={"Use Felzenszwalb"}
                isSelected={felzenszwalb}
                onValueChange={setFelzenszwalb}
              >
                Use Felzenszwalb
              </Checkbox>
              <Checkbox
                aria-label={"Use DDIM"}
                isSelected={ddim}
                onValueChange={setDdim}
              >
                Use DDIM
              </Checkbox>
            </div>
          </>
        )}
      </PopoverContent>
    </Popover>
  );
};

export default ModelParameterPicker;
