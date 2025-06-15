// const useSse = (
//   url: string,
//   onMessage: (data: any) => void = (_) => {},
//   onError: (error: Event) => void = () => {},
// ) => {
//   const [eventSource, setEventSource] = useState<EventSource | null>(null);
//
//   return {
//     start: useCallback(() => {
//       if (eventSource) {
//         eventSource.close();
//       }
//       const baseUrl = new URL(Constants.API_URL);
//       const apiUrl = new URL(url, baseUrl);
//       const tmp = new EventSource(apiUrl);
//
//       tmp.onmessage = (event) => onMessage(JSON.parse(event.data));
//       tmp.onerror = onError;
//
//       setEventSource(tmp);
//     }, [url, onMessage, onError, eventSource]),
//     stop: useCallback(() => {
//       if (eventSource) {
//         eventSource.close();
//         setEventSource(null);
//       }
//     }, [eventSource]),
//   };
// };

import { useCallback } from "react";

import useApi, { HTMLMethodRaw } from "@/hooks/use-api.ts";
import Constants from "@/constants.ts";
import { toCamelCase } from "@/utils/api-conversion.ts";

const useSse = (
  url: string,
  method: HTMLMethodRaw,
  body: any = null,
  onMessage: (data: any) => void = (_) => {},
  onError: (error: any) => void = () => {},
) => {
  const api = useApi();

  return useCallback(() => {
    const apiUrl = new URL(url, Constants.API_URL);
    const { controller, promise } = api[method](apiUrl.toString(), body);

    promise
      .then(async (response) => {
        const decoder = new TextDecoder("utf-8");
        const reader = response.body?.getReader();

        if (reader === undefined) {
          throw new Error("No reader");
        }

        while (true) {
          try {
            const { value, done } = await reader.read();

            if (done) break;
            onMessage(toCamelCase(JSON.parse(decoder.decode(value))));
          } catch (e) {
            onError(e);
            break;
          }
        }
      })
      .catch(onError);

    return controller.abort;
  }, [url, onMessage, onError]);
};

export default useSse;
