import { useCallback } from "react";

import Constants from "@/constants.ts";
import { toSnakeCase } from "@/utils/api-conversion.ts";

const methods = ["get", "post", "put", "delete", "patch"] as const;
const methodsRaw = methods.map((method) => `${method}Raw` as const);

export type HTMLMethod = (typeof methods)[number];
export type HTMLMethodRaw = (typeof methodsRaw)[number];

const useMethodCall = <Decode extends boolean = true>(
  method: HTMLMethod,
  decode: Decode,
) =>
  useCallback((url: string, data: any = null) => {
    const controller = new AbortController();
    const signal = controller.signal;

    const baseUrl = new URL(Constants.API_URL);
    const apiUrl = new URL(url, baseUrl);

    const promise = fetch(apiUrl, {
      method,
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: data === null ? null : JSON.stringify(toSnakeCase(data)),
      signal,
    }).then(async (response) => {
      return (
        decode
          ? {
              data: await response.json(),
              response: response,
            }
          : response
      ) as Decode extends true ? { data: any; response: Response } : Response;
    });

    return { promise, controller };
  }, []);

const useApi = () => {
  return {
    get: useMethodCall("get", true),
    post: useMethodCall("post", true),
    put: useMethodCall("put", true),
    delete: useMethodCall("delete", true),
    patch: useMethodCall("patch", true),
    getRaw: useMethodCall("get", false),
    postRaw: useMethodCall("post", false),
    putRaw: useMethodCall("put", false),
    deleteRaw: useMethodCall("delete", false),
    patchRaw: useMethodCall("patch", false),
  };
};

export default useApi;
