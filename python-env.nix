{ pkgs, python }:
let
  pythonPackages = python.pkgs;
in
pkgs.stdenv.mkDerivation rec {
  name = "python-env";
  src = ./.;

  # Separate output for the activation script
  activationScript = pkgs.runCommand "activate-venv-script" { } ''
    mkdir -p $out/bin
    cat > $out/bin/activate-venv <<EOF
    #!/bin/sh
    # Activate the virtual environment
    source ${venv}/bin/activate
    EOF
    chmod +x $out/bin/activate-venv
  '';

  venv = pkgs.stdenv.mkDerivation {
    name = "open-etl-venv";
    buildInputs = [ python ];

    phases = [ "installPhase" ];

    installPhase = ''
      # Create virtual environment using standard python venv
      ${python}/bin/python -m venv "$out"

      # Activate the virtual environment
      source "$out/bin/activate"

      # Install packages using pip
      "$out/bin/python" -m pip install --no-cache-dir \
        "tqdm">=4.0 \
        "pydantic>=2.5,<3" \
        "numpy>=1.26,<2" \
        "pandas>=2.1,<3" \
        "python-dotenv>=1,<2" \
        "requests>=2.31,<3" \
        "pytest>=8.3.4" \
        "setuptools==75.8.0" \

      deactivate
    '';
  };

  # Create an attribute set to hold both outputs
  out = {
    inherit venv activationScript;
  };
}