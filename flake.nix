# flake.nix
{
  description = "Open ETL Agent";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11"; # Ensure this nixpkgs version has Python 3.11
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix.url = "github:nix-community/poetry2nix"; # Add poetry2nix input
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }: # Add poetry2nix to function args
    flake-utils.lib.eachDefaultSystem (system:
      let
        # Base pkgs set
        pkgs = nixpkgs.legacyPackages.${system};

        # Create an environment using poetry2nix.
        # It reads pyproject.toml from projectDir by default.
        myPythonEnv = pkgs.poetry2nix.mkPoetryEnv {
          projectDir = ./.;                     # Location of pyproject.toml
          python = pkgs.python311;             # Base Python interpreter
          # If poetry2nix has trouble with a specific package, you might need overrides:
          # overrides = pkgs.poetry2nix.overrides.withDefaults (final: prev: {
          #   # Example: Override numpy if needed
          #   # numpy = prev.numpy.overridePythonAttrs (old: {
          #   #   doCheck = false; # Disable tests if they fail in nix build
          #   # });
          # });
        };
      in
      {
        devShells.default = pkgs.mkShell {
          # Use inputsFrom to get python interpreter and packages from mkPoetryEnv
          inputsFrom = [ myPythonEnv ];

          # Add other non-python tools if needed
          packages = [
             # e.g. pkgs.git
          ];

          shellHook = ''
            # Unset PYTHONPATH potentially set by user or other tools,
            # poetry2nix environment usually handles paths correctly.
            unset PYTHONPATH

            # Source local .env file if it exists
            if [ -f ".env" ]; then
              echo "Sourcing local .env file..."
              set -a # Automatically export all variables defined in .env
              source .env
              set +a
            fi

            echo ""
            echo "Nix development environment (using poetry2nix) ready."
            echo "Python interpreter with packages from pyproject.toml is available."
            # No manual venv activation needed! The python command is the correct one.
          '';
        };

        # Optional: Expose the derivation for the environment itself
        packages.pythonEnv = myPythonEnv;

        # Optional: If your project is a installable application
        # packages.default = pkgs.poetry2nix.mkPoetryApplication {
        #  projectDir = ./.;
        #  python = pkgs.python311;
        # };
      }
    );
}