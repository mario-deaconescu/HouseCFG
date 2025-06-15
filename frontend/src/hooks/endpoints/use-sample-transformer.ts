import { useCallback, useMemo, useState } from "react";

import { PlaceHolderPlan, Plan } from "@/types/serializers/plan.ts";
import useSse from "@/hooks/use-sse.ts";
import useToast from "@/hooks/use-toast.ts";

const useSampleTransformer = (graph: PlaceHolderPlan, num_steps: number) => {
  const [plan, setPlan] = useState<Plan | null>(null);
  const [isFinal, setIsFinal] = useState<boolean | null>(false);
  const [step, setStep] = useState<number>(0);
  const { errorToast } = useToast();
  const progress = useMemo(
    () => (isFinal ? 100 : ((step - 1) / num_steps) * 100),
    [step, isFinal, num_steps],
  );

  const sse = useSse(
    "generate/sample_transformer",
    "postRaw",
    {
      input: graph,
      num_steps: num_steps,
    },
    (data) => {
      if (!data["sample"]) {
        errorToast("Could not generate plan");
        setPlan(null);
        setIsFinal(null);
        setStep(0);

        return;
      }
      setPlan(data["sample"]);
      setIsFinal(data["final"]);
      setStep((prev) => prev + 1);
    },
    (_error) => {
      errorToast("Could not generate plan");
      setPlan(null);
      setIsFinal(null);
      setStep(0);
    },
  );

  const startSample = useCallback(() => {
    setStep(0);
    sse();
  }, [sse]);

  return { plan, isFinal, step, startSample, progress };
};

export default useSampleTransformer;
