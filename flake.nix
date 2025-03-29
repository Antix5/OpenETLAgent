# flake.nix
{
  description = "Open ETL Agent";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11"; # Ensure this nixpkgs version has Python 3.11
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix.url = "github:nix-community/poetry2nix";
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # --- CORRECTED PART ---
        # Import nixpkgs *with* the poetry2nix overlay applied.
        # This makes poetry2nix functions available directly under `pkgs`.
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ poetry2nix.overlays.default ]; # Apply the overlay here
        };
        # --- END CORRECTED PART ---

        # Now 'pkgs' contains poetry2nix attributes, so this line should work:
        myPythonEnv = pkgs.poetry2nix.mkPoetryEnv {
          projectDir = ./.;
          python = pkgs.python311; # Use the overlaid pkgs consistently
          # overrides = pkgs.poetry2nix.overrides.withDefaults (final: prev: { ... });
        };

      in
      {
        devShells.default = pkgs.mkShell { # Use the overlaid pkgs here too
          inputsFrom = [ myPythonEnv ];
          packages = [
             # Add other tools using the overlaid pkgs, e.g.: pkgs.git
          ];
          shellHook = ''
            unset PYTHONPATH
            if [ -f ".env" ]; then
              echo "Sourcing local .env file..."
              set -a; source .env; set +a
            fi
            echo ""
            echo "Nix development environment (using poetry2nix) ready."
            echo "Python interpreter with packages from pyproject.toml is available."
          '';
        };

        packages.pythonEnv = myPythonEnv;
        # packages.default = pkgs.poetry2nix.mkPoetryApplication { ... }; # Use overlaid pkgs here too
      }
    );
}