import { PropsWithChildren } from "react";

declare module "react-query" {
  export interface QueryClientProviderProps extends PropsWithChildren<{}> {}
}
