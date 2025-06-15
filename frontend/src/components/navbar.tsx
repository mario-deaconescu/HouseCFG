import { Link } from "@heroui/link";
import {
  Navbar as HeroUINavbar,
  NavbarBrand,
  NavbarContent,
} from "@heroui/navbar";

export const Navbar = () => {
  return (
    <HeroUINavbar maxWidth="xl" position="sticky">
      <NavbarContent className="basis-1/5 sm:basis-full" justify="start">
        <NavbarBrand className="gap-3 max-w-fit">
          <Link
            className="flex justify-start items-center gap-1"
            color="foreground"
            href="/"
          >
            {/*<Logo />*/}
            <p className="font-bold text-inherit">HouseCFG</p>
          </Link>
        </NavbarBrand>
        {/*<div className="hidden lg:flex gap-4 justify-start ml-2">*/}
        {/*  {siteConfig.navItems.map((item) => (*/}
        {/*    <NavbarItem key={item.href}>*/}
        {/*      <Link*/}
        {/*        className={clsx(*/}
        {/*          linkStyles({ color: "foreground" }),*/}
        {/*          "data-[active=true]:text-primary data-[active=true]:font-medium",*/}
        {/*        )}*/}
        {/*        color="foreground"*/}
        {/*        href={item.href}*/}
        {/*      >*/}
        {/*        {item.label}*/}
        {/*      </Link>*/}
        {/*    </NavbarItem>*/}
        {/*  ))}*/}
        {/*</div>*/}
      </NavbarContent>
    </HeroUINavbar>
  );
};
