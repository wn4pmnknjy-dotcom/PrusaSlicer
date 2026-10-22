#include <algorithm>
#include <set>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <boost/filesystem.hpp>

#include "Slic3r/Biz/Preset/IO/BundleLoader.hpp"
#include "Slic3r/Biz/Preset/IO/BundlePaths.hpp"
#include "Slic3r/Biz/Preset/HwConfigEvaluator.hpp"
#include "Slic3r/Biz/Preset/PresetEvaluator.hpp"
#include "Slic3r/TestUtils/TestData.hpp"

namespace fs = boost::filesystem;

TEST_CASE("Bundled castable wax SLA presets load and evaluate", "[preset]")
{
    using namespace Slic3r::Biz::Preset;
    using namespace Slic3r::Biz::Preset::IO;
    using namespace Slic3r::Domain::Preset;

    // The presets shipped with the application, not the test data copy.
    const fs::path resources = Tests::get_datadir() / ".." / ".." / ".." / ".." / "resources";
    BundlePaths paths        = BundlePaths::make_test_runtime(Tests::get_datadir());
    paths.app_bundle_path    = fs::canonical(resources / "presets").string();

    const Bundle bundle = load_bundle(paths);

    const auto vendor_it = bundle.vendor_bundles.find("JewelryCastable");
    REQUIRE(vendor_it != bundle.vendor_bundles.end());
    const VendorBundle& vendor = vendor_it->second;

    // The application instantiates printer configs from the vendor templates.
    HwConfigEvaluator config_evaluator;
    PresetEvaluator evaluator{vendor.presets};
    for (const std::string id : {"msla2k", "msla4k"}) {
        const HwPrinterConfigTemplate* templ =
            vendor.vendor_data.find_printer_config_template_by_id(id);
        REQUIRE(templ != nullptr);
        const HwPrinterConfig hw_config =
            config_evaluator.create_printer_config(*templ, vendor.vendor_data);

        const auto printers = evaluator.evaluate(hw_config);
        REQUIRE(printers.size() == 1);
        const EvaluatedPrinterPreset& printer = printers.front();
        CHECK(printer.technology() == Slic3r::Domain::PrinterTechnology::SLA);

        std::set<std::string> print_names;
        for (const EvaluatedPrintPreset& print : printer.prints) {
            print_names.insert(print.preset.name);
            REQUIRE_FALSE(print.materials.empty());
            CHECK(
                std::ranges::any_of(
                    print.materials.front(),
                    [](const auto& m) { return m.preset.name.starts_with("Generic Castable Wax"); }
                )
            );
        }
        CHECK(print_names.contains("0.03 Castable Wax Fine"));
        CHECK(print_names.contains("0.05 Castable Wax"));
        CHECK(print_names.contains("0.05 Castable Wax Heavy"));
    }
}
