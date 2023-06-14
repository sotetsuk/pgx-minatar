# MinAtar extension for Pgx

https://www.gnu.org/licenses/gpl-faq.en.html#GPLPlugins

> When is a program and its plug-ins considered a single combined program? ([#GPLPlugins](https://www.gnu.org/licenses/gpl-faq.en.html#GPLPlugins))
It depends on how the main program invokes its plug-ins. If the main program uses fork and exec to invoke plug-ins, and they establish intimate communication by sharing complex data structures, or shipping complex data structures back and forth, that can make them one single combined program. **A main program that uses simple fork and exec to invoke plug-ins and does not establish intimate communication between them results in the plug-ins being a separate program.**


Note that [Pgx](https://github.com/sotetsuk/pgx) just invoke [`pgx-minatar`](https://github.com/sotetsuk/pgx-minatar) environments through `pgx.make` (e.g., `env = pgx.make("minatar-asterix")`).
All MinAtar environment dynamics are implemented in this MinAtar extension repository.
