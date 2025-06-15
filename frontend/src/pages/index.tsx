import { Card, CardBody } from "@heroui/card";
import { Tab, Tabs } from "@heroui/tabs";
import { Key, useCallback, useEffect, useMemo, useState } from "react";
import { FaEraser, FaPaintbrush } from "react-icons/fa6";
import { PiRectangleDashed, PiRectangleDashedDuotone } from "react-icons/pi";
import { Button, ButtonGroup } from "@heroui/button";
import { SharedSelection } from "@heroui/system";
import { Slider } from "@heroui/slider";
import { Select, SelectItem } from "@heroui/select";
import { NumberInput } from "@heroui/number-input";
import { addToast } from "@heroui/toast";
import { EventSourceParserStream } from "eventsource-parser/stream";
import { Progress } from "@heroui/progress";
import { Pagination } from "@heroui/pagination";

import DefaultLayout from "@/layouts/default";
import ModelPicker from "@/components/model-picker.tsx";
import { Model, useModel } from "@/context/model-context.tsx";
import MaskCanvas from "@/components/mask-canvas.tsx";
import { defaultMask, Mask, MaskMode, transpose } from "@/types/mask";
import { canvasSize } from "@/types/canvas.ts";
import BubbleCanvas from "@/components/bubble-canvas.tsx";
import { RoomType } from "@/types/room-type.ts";
import { Bubble, BubbleMask } from "@/types/bubble-mask.ts";
import {
  BaseInputParameters,
  BubblesInputParameters,
  generateBubbles,
  generateRoomTypes,
  RoomTypeInputParameters,
} from "@/api";
import SampleImage from "@/components/sample-image.tsx";
import { SampleList } from "@/types/sample.ts";
import { defaultMap, validRoomTypes } from "@/types/room-type-picker.ts";
import RoomTypePicker from "@/components/room-type-picker.tsx";
import ModelParameterPicker from "@/components/model-parameter-picker.tsx";
import {
  defaultModelParameters,
  ModelParameters,
} from "@/types/model-parameters.ts";
import SamplePlan from "@/components/sample-plan.tsx";

const maskCanvasModes = [
  {
    mode: MaskMode.BRUSH,
    icon: <FaPaintbrush />,
    size: "text-xl",
  },
  {
    mode: MaskMode.ERASER,
    icon: <FaEraser />,
    size: "text-2xl",
  },
  {
    mode: MaskMode.FILL,
    icon: <PiRectangleDashedDuotone />,
    size: "text-2xl",
  },
  {
    mode: MaskMode.FILL_ERASER,
    icon: <PiRectangleDashed />,
    size: "text-2xl",
  },
];

const roomTypeOptions = RoomType.entries()
  .map(([key, value]) => ({
    key: key,
    value: value,
    label: RoomType.keyToString(key),
  }))
  .filter(
    ({ value }) =>
      ![RoomType.FRONT_DOOR, RoomType.INTERIOR_DOOR, RoomType.UNKNOWN].includes(
        value,
      ),
  );

const steps = Array.from({ length: 1001 })
  .map((_, i) => i)
  .filter((value) => 1000 % value === 0);

enum ResultType {
  IMAGE = "image",
  ORIGINAL_IMAGE = "original_image",
  PLAN = "plan",
}

const resultTypeString = (type: ResultType) => {
  switch (type) {
    case ResultType.IMAGE:
      return "Image";
    case ResultType.ORIGINAL_IMAGE:
      return "Original";
    case ResultType.PLAN:
      return "Plan";
  }
};

export default function IndexPage() {
  const { model } = useModel();
  const [maskCanvasMode, setMaskCanvasMode] = useState<MaskMode>(
    MaskMode.BRUSH,
  );
  const [mask, setMask] = useState<Mask<boolean>>(
    defaultMask(canvasSize, canvasSize, false),
  );
  const [bubbles, setBubbles] = useState<Bubble[]>([]);
  const [activeTab, setActiveTab] = useState("mask");
  const [bubbleSize, setBubbleSize] = useState(1);
  const [bubbleRoomType, setBubbleRoomType] = useState<RoomType>(
    RoomType.BEDROOM,
  );
  const [progress, setProgress] = useState<number | null>(null);
  const [modelParameters, setModelParameters] = useState<ModelParameters>(
    defaultModelParameters,
  );
  const [numSamples, setNumSamples] = useState(1);
  const [numStepsSlider, setNumStepsSlider] = useState(10);
  const [sampleList, setSampleList] = useState<SampleList | null>(null);
  const [currentSampleIndex, setCurrentSampleIndex] = useState(0);
  const [abortController, setAbortController] =
    useState<AbortController | null>(null);
  const [roomTypes, setRoomTypes] = useState<Map<RoomType, number>>(defaultMap);
  const [resultType, setResultType] = useState<ResultType>(
    ResultType.ORIGINAL_IMAGE,
  );

  const numSteps = useMemo(() => steps[numStepsSlider], [numStepsSlider]);

  const selectBubbleRoomType = useCallback(
    (selection: SharedSelection) => {
      if (!(selection instanceof Set) || selection.size !== 1) {
        throw new Error("Invalid selection");
      }

      const value = selection.values().next().value;

      if (value === undefined) {
        throw new Error("Invalid selection");
      }

      setBubbleRoomType(value as RoomType);
    },
    [setBubbleRoomType],
  );

  const selectedBubbleRoomType = useMemo(() => {
    return new Set([bubbleRoomType.toString()]);
  }, [bubbleRoomType]);

  const generatePromise = useCallback(async () => {
    setProgress(0);
    setResultType(ResultType.ORIGINAL_IMAGE);

    const globalMask = transpose(mask);
    const baseParams: BaseInputParameters = {
      ...modelParameters,
      num_samples: numSamples,
      num_steps: numSteps,
      mask: [globalMask],
      as_image: true,
    };
    let fullParams: BubblesInputParameters | RoomTypeInputParameters;
    let func;

    if (model === Model.BUBBLES) {
      const bubbleMask = transpose(new BubbleMask(bubbles).toMask());

      func = generateBubbles;
      fullParams = {
        ...baseParams,
        bubbles: [bubbleMask],
      };
    } else {
      let roomTypeVector: number[] = new Array<number>(
        RoomType.restrictedLength(),
      );

      validRoomTypes.map(([_, value]) => {
        const index = RoomType.toRestricted(value);

        console.log(value, index);

        roomTypeVector[index] = roomTypes.get(value) ?? 0;
      });
      roomTypeVector[RoomType.toRestricted(RoomType.UNKNOWN)] = 0;

      console.log(roomTypeVector);
      func = generateRoomTypes;
      fullParams = {
        ...baseParams,
        room_types: roomTypeVector,
      };
    }

    try {
      const controller = new AbortController();

      setAbortController(controller);
      const response = await func({
        body: fullParams,
        parseAs: "stream",
        signal: controller.signal,
      });

      const data = response.data as ReadableStream | undefined;

      if (!data) {
        throw new Error("No data");
      }
      const eventStream = data
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(new EventSourceParserStream());

      const reader = eventStream.getReader();

      let sampleId = 1;

      while (true) {
        const { done, value } = await reader.read();

        if (done || value === undefined) {
          break;
        }
        let chunk = value.data;

        const data = JSON.parse(chunk) as SampleList;

        if (data) {
          setSampleList(data);
          setProgress((sampleId++ / numSteps) * 100);
        }
      }

      setResultType(ResultType.IMAGE);
    } catch (error) {
      addToast({
        severity: "danger",
        title: "Error",
        description: "An error occurred while generating the plan.",
      });
      console.error("Error generating plan:", error);
    } finally {
      setAbortController(null);
      setProgress(null);
    }
  }, [
    mask,
    bubbles,
    model,
    modelParameters,
    numSamples,
    numSteps,
    roomTypes,
    resultType,
  ]);

  const cancelGeneration = useCallback(() => {
    if (abortController) {
      abortController.abort();
      setProgress(null);
      setAbortController(null);
    }
  }, [abortController]);

  const progressBar = useMemo(
    () =>
      progress !== null && (
        <Progress
          aria-label="Generating..."
          className="flex-grow"
          value={progress}
        />
      ),
    [progress],
  );

  const generate = useCallback(() => {
    addToast({
      title: "Generating Plan",
      description: "This may take a while",
      promise: generatePromise(),
      timeout: 1,
    });
  }, [generatePromise, progressBar]);

  useEffect(() => {
    setActiveTab("mask");
  }, [model]);

  return (
    <DefaultLayout>
      <div
        className={"max-h-[90vh] max-w-[110vh] h-full flex flex-col mx-auto"}
      >
        <div className={"flex flex-row space-between mb-3 items-center"}>
          <p className={"flex-grow text-2xl font-bold"}> Generate Plan </p>
          <div className={"flex flex-row gap-5 items-center"}>
            <div className={" w-[20rem]"}>
              <ModelPicker />
            </div>
          </div>
        </div>
        <Card
          className={"flex-shrink h-fit"}
          classNames={{
            body: "h-full",
          }}
        >
          <CardBody className={"flex flex-row gap-2 items-start flex-shrink"}>
            <div
              className={"w-1/2 relative"}
              onWheel={(e) => {
                setBubbleSize((prevSize) => {
                  const newSize = prevSize + (e.deltaY < 0 ? 1 : -1);

                  return Math.max(1, Math.min(newSize, canvasSize));
                });
              }}
            >
              <Tabs
                classNames={{
                  base: "mb-3",
                  panel: "p-0",
                }}
                destroyInactiveTabPanel={false}
                selectedKey={activeTab}
                onSelectionChange={(key: Key) => setActiveTab(key as string)}
              >
                <Tab key={"mask"} title={"Mask"}>
                  <Card classNames={{ base: "aspect-square" }}>
                    <CardBody className={"p-0"}>
                      <MaskCanvas
                        className={"w-full h-full"}
                        mode={maskCanvasMode}
                        scaleFactor={15}
                        onChange={setMask}
                      />
                    </CardBody>
                  </Card>
                </Tab>
                {model === Model.BUBBLES && (
                  <Tab key={"bubbles"} title={"Bubble Diagram"}>
                    <Card classNames={{ base: "aspect-square" }}>
                      <CardBody className={"p-0"}>
                        <BubbleCanvas
                          backgroundMask={mask}
                          className={"w-full h-full"}
                          scaleFactor={15}
                          selectedRoomType={bubbleRoomType}
                          size={bubbleSize / 2}
                          onChange={setBubbles}
                        />
                      </CardBody>
                    </Card>
                  </Tab>
                )}
                {model === Model.ROOM_TYPES && (
                  <Tab key={"room-types"} title={"Room Types"}>
                    <RoomTypePicker onChange={setRoomTypes} />
                  </Tab>
                )}
              </Tabs>
              <div className={"h-12 mt-3 flex flex-row gap-5 items-center"}>
                {activeTab === "mask" && (
                  <ButtonGroup>
                    {maskCanvasModes.map((item) => (
                      <Button
                        key={item.mode}
                        isIconOnly
                        className={item.size}
                        color={
                          item.mode === maskCanvasMode ? "primary" : "default"
                        }
                        onPress={() => {
                          setMaskCanvasMode(item.mode);
                        }}
                      >
                        {item.icon}
                      </Button>
                    ))}
                  </ButtonGroup>
                )}
                {activeTab === "bubbles" && (
                  <>
                    <Select
                      className={"w-[10rem]"}
                      defaultSelectedKeys={selectedBubbleRoomType}
                      label={"Select Room Type"}
                      selectedKeys={selectedBubbleRoomType}
                      onSelectionChange={selectBubbleRoomType}
                    >
                      {roomTypeOptions.map((option) => (
                        <SelectItem key={option.value}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </Select>
                    <Slider
                      aria-label={"Bubble Size"}
                      className="w-[15rem]"
                      defaultValue={1}
                      endContent={<p className={"text-nowrap"}>Bubble Size</p>}
                      maxValue={canvasSize}
                      minValue={1}
                      showTooltip={true}
                      step={1}
                      value={bubbleSize}
                      onChange={(value) =>
                        typeof value == "number" && setBubbleSize(value)
                      }
                    />
                  </>
                )}
              </div>
            </div>
            <div className={"w-1/2"}>
              <div
                className={
                  "h-10 mb-3 flex flex-row justify-end items-center gap-5"
                }
              >
                {sampleList !== null && (
                  <>
                    <Pagination
                      className={""}
                      showControls={true}
                      total={sampleList.images.length}
                      onChange={(value) => setCurrentSampleIndex(value - 1)}
                    />
                    <Select
                      className={"w-[8rem]"}
                      label={"Show Result"}
                      selectedKeys={[resultType]}
                      size={"sm"}
                      onSelectionChange={(value) => {
                        if (value instanceof Set) {
                          setResultType(
                            (value.values().next().value as ResultType) ??
                              ResultType.ORIGINAL_IMAGE,
                          );
                        }
                      }}
                    >
                      {Object.entries(ResultType).map(([_, value]) => (
                        <SelectItem key={value}>
                          {resultTypeString(value)}
                        </SelectItem>
                      ))}
                    </Select>
                  </>
                )}
                <Button
                  color={"primary"}
                  isLoading={progress !== null}
                  onPress={generate}
                >
                  Generate
                </Button>
              </div>
              <Card classNames={{ base: "aspect-square" }}>
                <CardBody className={"p-0"}>
                  {sampleList !== null &&
                    (resultType === ResultType.PLAN ? (
                      sampleList.plans[currentSampleIndex] !== null && (
                        <SamplePlan
                          plan={sampleList.plans[currentSampleIndex]}
                          scaleFactor={15}
                        />
                      )
                    ) : (
                      <SampleImage
                        canvasSize={
                          resultType === ResultType.ORIGINAL_IMAGE ? 64 : 256
                        }
                        className={"w-full h-full"}
                        removeBackground={
                          resultType !== ResultType.ORIGINAL_IMAGE
                        }
                        sample={{
                          image: (resultType === ResultType.ORIGINAL_IMAGE
                            ? sampleList.original_images
                            : sampleList.images)[currentSampleIndex],
                        }}
                        scaleFactor={15}
                      />
                    ))}
                </CardBody>
              </Card>
              <div
                className={
                  "h-12 mt-3 flex flex-row gap-5 items-center justify-end"
                }
              >
                {progressBar ? (
                  <>
                    {progressBar}
                    <Button color={"danger"} onPress={cancelGeneration}>
                      Cancel
                    </Button>
                  </>
                ) : (
                  <>
                    <NumberInput
                      className={"w-[7rem] flex-grow"}
                      label={"Num Samples"}
                      value={numSamples}
                      onValueChange={setNumSamples}
                    />
                    <Slider
                      aria-label={"Num Steps"}
                      className="w-[7rem] flex-grow"
                      defaultValue={100}
                      label={"Num Steps"}
                      maxValue={steps.length - 1}
                      minValue={0}
                      renderValue={() => numSteps}
                      showTooltip={false}
                      step={1}
                      value={numStepsSlider}
                      onChange={(value) =>
                        typeof value == "number" && setNumStepsSlider(value)
                      }
                    />
                    <ModelParameterPicker
                      values={modelParameters}
                      onChange={setModelParameters}
                    />
                  </>
                )}
              </div>
            </div>
          </CardBody>
        </Card>
      </div>
    </DefaultLayout>
  );
}
